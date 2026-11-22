"""
Evaluate Hook

Author: Xuweiyi Chen (xuweic@email.virginia.edu)
        Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import pointops

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, save_pred_dir=None):
        # If set, dumps per-scene .npz {coord, segment, pred} at original
        # resolution under cfg.save_path/<save_pred_dir>/. Single-GPU only:
        # multi-GPU evaluation shards the dataset across ranks, which breaks
        # the scene-name lookup used here.
        self.save_pred_dir = save_pred_dir

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        save_pred_root = None
        if self.save_pred_dir is not None:
            if comm.get_world_size() > 1:
                self.trainer.logger.warning(
                    f"[SemSegEvaluator] save_pred_dir is set but world_size>1; "
                    f"per-rank dataset sharding makes the name lookup unsafe — "
                    f"skipping prediction dump. Re-run on 1 GPU to save preds."
                )
            else:
                save_pred_root = os.path.join(self.trainer.cfg.save_path, self.save_pred_dir)
                os.makedirs(save_pred_root, exist_ok=True)
                self.trainer.logger.info(f"[SemSegEvaluator] dumping per-scene preds → {save_pred_root}")

        sample_cursor = 0  # running global sample index for save_pred naming
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                # Map the voxel predictions to every original point (1-NN).
                if "origin_offset" not in input_dict.keys():
                    # Infer from offset ratio: origin has more points per sample
                    # Each sample's origin_coord count = origin_segment count
                    n_origin = input_dict["origin_segment"].shape[0]
                    n_down = input_dict["segment"].shape[0]
                    offset = input_dict["offset"]
                    # For single sample or proportional split
                    if len(offset) == 1:
                        origin_offset = torch.tensor([n_origin], device=offset.device, dtype=offset.dtype)
                    else:
                        # Proportional: origin_count_i / total_origin = down_count_i / total_down
                        ratios = offset.float() / n_down
                        origin_offset = (ratios * n_origin).long()
                        origin_offset[-1] = n_origin  # ensure last matches
                    input_dict["origin_offset"] = origin_offset
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            if save_pred_root is not None:
                # Split the batch by per-sample offset so that a batch with
                # several samples saves one .npz per scene under its own name.
                ds = self.trainer.val_loader.dataset
                save_coord_full = (
                    input_dict["origin_coord"]
                    if "origin_coord" in input_dict
                    else input_dict["coord"]
                ).detach().cpu().numpy().astype(np.float32)
                save_seg_full = segment.detach().cpu().numpy().astype(np.int32)
                save_pred_full = pred.detach().cpu().numpy().astype(np.int32)

                # Sample boundaries — use origin_offset if we worked at origin
                # resolution, else offset.
                if "origin_coord" in input_dict and "origin_offset" in input_dict:
                    boundaries = input_dict["origin_offset"].detach().cpu().numpy().tolist()
                else:
                    boundaries = input_dict["offset"].detach().cpu().numpy().tolist()
                starts = [0] + boundaries[:-1]
                ends = boundaries

                # Optional per-point frame_idx (origin_frame_idx if present,
                # else post-voxel frame_idx — only emit if it matches the
                # length of the saved coord array).
                fi_full = None
                for src_key in ("origin_frame_idx", "frame_idx"):
                    if src_key in input_dict:
                        cand = input_dict[src_key].detach().cpu().numpy().astype(np.int32)
                        if cand.shape[0] == save_coord_full.shape[0]:
                            fi_full = cand
                            break

                for j, (s, e) in enumerate(zip(starts, ends)):
                    global_idx = sample_cursor + j
                    try:
                        scene_name = ds.get_data_name(global_idx)
                    except Exception:
                        scene_name = f"scene_{global_idx:04d}"
                    extra = {}
                    if fi_full is not None:
                        extra["frame_idx"] = fi_full[s:e]
                    np.savez_compressed(
                        os.path.join(save_pred_root, f"{scene_name}.npz"),
                        coord=save_coord_full[s:e],
                        segment=save_seg_full[s:e],
                        pred=save_pred_full[s:e],
                        **extra,
                    )

                sample_cursor += len(starts)

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)

        # Log to wandb if a run is active (main process only)
        if wandb is not None and wandb.run is not None and comm.is_main_process():
            wandb.log({
                "val/loss": loss_avg,
                "val/mIoU": m_iou,
                "val/mAcc": m_acc,
                "val/allAcc": all_acc,
                "epoch": current_epoch
            })

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        if np.isfinite(self.trainer.best_metric_value):  # only set when checkpoints are saved during training
            self.trainer.logger.info(
                "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
            )
