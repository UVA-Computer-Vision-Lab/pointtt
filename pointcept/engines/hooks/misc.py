"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

Modified for PoinTTT: the checkpoint saver keeps every epoch's checkpoint by default
(`save_every_epoch = False` keeps every `save_freq`-th one), writes checkpoints atomically and
stores the best metric as a Python float; the checkpoint loader uses
torch.load(weights_only=True), also for resuming, and fails on a missing weight file; no
TensorBoard logging.
"""

import os
import time
from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from pointcept.utils.comm import is_main_process

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._last_time = time.perf_counter()
        self._remain_iter = 0

    def before_train(self):
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._last_time = time.perf_counter()

    def before_step(self):
        data_time = time.perf_counter() - self._last_time
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = time.perf_counter() - self._last_time
        self._last_time = time.perf_counter()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""

    def before_step(self):
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)


def _atomic_copy(src, dst):
    """Copy src to dst through a temporary file, so that dst is never half-written."""
    with open(src, "rb") as f_in, open(dst + ".tmp", "wb") as f_out:
        while True:
            buf = f_in.read(64 << 20)
            if not buf:
                break
            f_out.write(buf)
    os.replace(dst + ".tmp", dst)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    """Saves model/epoch_N.pth after every epoch, mirrors it to model_last.pth, and copies it
    to model_best.pth whenever the evaluator reports a new best metric.

    With `save_every_epoch = False` in the config, epoch_N.pth is kept only every `save_freq`
    epochs; model_last.pth and model_best.pth are always written.
    """

    def __init__(self, save_freq):
        self.save_freq = save_freq

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                # set by SemSegEvaluator, which runs before this hook
                current_metric_value = float(self.trainer.comm_info["current_metric_value"])
                current_metric_name = self.trainer.comm_info["current_metric_name"]

                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            epoch_filename = os.path.join(
                self.trainer.cfg.save_path, "model", f"epoch_{self.trainer.epoch + 1}.pth"
            )
            last_filename = os.path.join(self.trainer.cfg.save_path, "model", "model_last.pth")
            keep_epoch = (
                getattr(self.trainer.cfg, "save_every_epoch", True)
                or (self.trainer.epoch + 1) % self.save_freq == 0
            )
            if not keep_epoch:
                epoch_filename = last_filename
            self.trainer.logger.info("Saving checkpoint to: " + epoch_filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": float(self.trainer.best_metric_value),
                },
                epoch_filename + ".tmp",
            )
            os.replace(epoch_filename + ".tmp", epoch_filename)
            if keep_epoch:
                _atomic_copy(epoch_filename, last_filename)
            if is_best:
                _atomic_copy(
                    epoch_filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, strict=False):
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            # Checkpoints hold tensors and plain Python values only (training checkpoints
            # also the optimizer, scheduler and AMP-scaler states), so nothing else needs
            # to be unpickled.
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=True,
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = float(checkpoint["best_metric_value"])
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        elif self.trainer.cfg.weight:
            raise FileNotFoundError(
                f"cfg.weight is set but the file does not exist: {self.trainer.cfg.weight}"
            )
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
