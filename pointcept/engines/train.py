"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

TrainerBase follows the hook-based training loop of detectron2 (Apache-2.0).
Modified for PoinTTT: optional W&B logging (`enable_wandb`), evaluation-only runs
(`is_train = False`), gradient accumulation (`grad_accum_steps`), support for models that
run backward() inside their forward pass (StreamingSegmentorV2), and single-process
reproductions of multi-GPU runs (`ddp_emulate_world_size`, `ddp_replay_world_size`).
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

try:
    import wandb
except ImportError:
    wandb = None

TRAINERS = Registry("trainers")


class ReplayRanksSampler(torch.utils.data.distributed.DistributedSampler):
    """One process drawing, in order, the batches of every rank of a DDP run.

    Loader batch i (batch_size = per-rank batch) is the batch that rank
    i % num_replicas drew at iteration i // num_replicas of a
    num_replicas-process run (DistributedSampler + DataLoader(drop_last=True)).
    With grad_accum_steps = num_replicas, one optimizer step then sees the same
    samples, in the same per-rank groups, as one step of that run, and an epoch
    has the same number of optimizer steps.
    """

    def __init__(self, dataset, num_replicas, batch_size):
        super().__init__(dataset, num_replicas=num_replicas, rank=0)
        self.batch_size = batch_size
        self.num_iters = self.num_samples // batch_size  # per rank, drop_last

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * (padding_size // len(indices) + 1))[:padding_size]
        w, b = self.num_replicas, self.batch_size
        return iter(
            [
                indices[r + w * (k * b + j)]
                for k in range(self.num_iters)
                for r in range(w)
                for j in range(b)
            ]
        )

    def __len__(self):
        return self.num_iters * self.num_replicas * self.batch_size


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        if self.cfg.is_train:
            self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        if self.cfg.is_train:
            self.optimizer = self.build_optimizer()
            self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        steps_per_epoch = len(self.train_loader)
        replay = int(getattr(self.cfg, "ddp_replay_world_size", 0) or 0)
        if replay > 1 and comm.get_world_size() == 1:
            # replay mode (see build_train_loader): run_step() steps the scheduler
            # once per `replay` loader iterations, like the run it replays.
            steps_per_epoch //= replay
            self.logger.info(
                f"Scheduler: {steps_per_epoch} optimizer steps/epoch x "
                f"{self.cfg.eval_epoch} epochs = {steps_per_epoch * self.cfg.eval_epoch} total steps"
            )
        self.cfg.scheduler.total_steps = steps_per_epoch * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def init_wandb(self):
        """Optional Weights & Biases logging, enabled with `enable_wandb = True` in the config."""
        if not (comm.is_main_process() and getattr(self.cfg, "enable_wandb", False)):
            return
        if wandb is None:
            self.logger.warning("enable_wandb is set but wandb is not installed: W&B logging disabled.")
            return
        try:
            wandb.init(
                project=getattr(self.cfg, "project_name", "pointtt"),
                name=getattr(self.cfg, "experiment_name", "default_experiment"),
            )
        except Exception as e:
            # no W&B login / network on this machine: keep going without W&B
            self.logger.warning(
                f"wandb.init failed ({type(e).__name__}: {e}); "
                "W&B logging disabled for this run."
            )

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            self.init_wandb()

            if self.cfg.is_train:
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    # => before epoch
                    # (also in one process when the sampler emulates or replays DDP ranks)
                    if comm.get_world_size() > 1 or isinstance(
                        self.train_loader.sampler,
                        torch.utils.data.distributed.DistributedSampler,
                    ):
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    # => run_epoch
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        # => before_step
                        self.before_step()
                        # => run_step
                        self.run_step()
                        # => after_step
                        self.after_step()
                    # => after epoch
                    self.after_epoch()
            else:
                # evaluation only: run the evaluation hooks once
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        accum_steps = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        cur_iter = self.comm_info["iter"]
        last_iter = cur_iter == len(self.train_loader) - 1
        is_first_micro = (cur_iter % accum_steps) == 0
        is_update_step = ((cur_iter + 1) % accum_steps == 0) or last_iter

        with torch.amp.autocast(device_type="cuda", enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        # StreamingSegmentorV2 runs backward() chunk by chunk inside its forward pass and
        # returns a detached loss flagged with `_skip_backward`.
        skip_backward = getattr(loss, "_skip_backward", False)
        if (
            skip_backward
            and comm.get_world_size() > 1
            and getattr(self.cfg, "streaming_grad_sync", False)
        ):
            self._average_grads_across_ranks()
        if not skip_backward and accum_steps > 1:
            loss = loss / accum_steps
        if not skip_backward and is_first_micro:
            self.optimizer.zero_grad()
        grad_norm = None
        if self.cfg.enable_amp:
            if not skip_backward:
                self.scaler.scale(loss).backward()
            if is_update_step:
                if not skip_backward:
                    self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                if not skip_backward:
                    self.scaler.step(self.optimizer)
                    scaler = self.scaler.get_scale()
                    self.scaler.update()
                    if scaler <= self.scaler.get_scale():
                        self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
        else:
            if not skip_backward:
                loss.backward()
            if is_update_step:
                if self.cfg.clip_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
        if wandb is not None and wandb.run is not None and comm.is_main_process():
            wandb.log(
                {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item() if grad_norm is not None else None,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                }
            )

    def _average_grads_across_ranks(self):
        """Average `.grad` over all DDP ranks (what DDP normally does itself).

        StreamingSegmentorV2 calls backward() inside forward() and returns a
        detached loss, so DDP's reducer never sees a backward pass it can hook
        into. With `streaming_grad_sync = True` the gradients are averaged here
        instead; off by default.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        had_grad = [p.grad is not None for p in params]
        flat = torch.cat(
            [
                (p.grad if g else torch.zeros_like(p)).reshape(-1).float()
                for p, g in zip(params, had_grad)
            ]
        )
        # number of ranks holding a grad per parameter: one that has a grad on
        # any rank gets the same averaged grad on every rank (as DDP does)
        n_ranks = torch.tensor(had_grad, dtype=torch.float32, device=flat.device)
        dist.all_reduce(flat)
        dist.all_reduce(n_ranks)
        flat /= comm.get_world_size()
        offset = 0
        for p, g, any_rank in zip(params, had_grad, n_ranks.tolist()):
            n = p.numel()
            avg = flat[offset : offset + n].view_as(p)
            offset += n
            if g:
                p.grad.copy_(avg)
            elif any_rank > 0:
                p.grad = avg.to(p.dtype).clone()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        ddp_kwargs = dict(
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        # static_graph=True is required when the wrapped module calls
        # backward() multiple times per forward (e.g. chunk-wise per-chunk
        # backward in PT-v3-LaCT-Dec / PT-v3-LaCT-Temporal). Without it DDP
        # raises "Parameter at index N has been marked as ready twice."
        if getattr(self.cfg, "ddp_static_graph", False):
            ddp_kwargs["static_graph"] = True
        model = create_ddp_model(model.cuda(), **ddp_kwargs)
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        emulate = int(getattr(self.cfg, "ddp_emulate_world_size", 0) or 0)
        replay = int(getattr(self.cfg, "ddp_replay_world_size", 0) or 0)
        accum_steps = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        if comm.get_world_size() > 1:
            # a run started in replay mode keeps both keys in its config.py
            assert replay <= 1 or accum_steps == 1, (
                f"this run was started with ddp_replay_world_size={replay}: "
                "continue it under real DDP with grad_accum_steps=1"
            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        elif replay > 1:
            # One process that draws the batches of all `replay` ranks in turn;
            # with grad_accum_steps = replay each optimizer step is the step of
            # the `replay`-process run (see ReplayRanksSampler).
            assert accum_steps == replay, (
                f"ddp_replay_world_size={replay} needs grad_accum_steps={replay} "
                f"(got {accum_steps})"
            )
            train_sampler = ReplayRanksSampler(
                train_data, replay, self.cfg.batch_size_per_gpu
            )
        elif emulate > 1:
            # One process that sees exactly the data stream of rank 0 in an
            # `emulate`-process run (same shard of every epoch's permutation,
            # same number of iterations per epoch). A StreamingSegmentorV2 run
            # without `streaming_grad_sync` never averages gradients across
            # ranks, so this process trains exactly like rank 0 of that run.
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, num_replicas=emulate, rank=0
            )
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_scaler(self):
        # replay mode: run_step() divides the loss by grad_accum_steps (= replay)
        # before scaling it; starting the scale that much higher keeps
        # scale * loss of a micro-batch what each rank of the replayed run had.
        replay = int(getattr(self.cfg, "ddp_replay_world_size", 0) or 0)
        init_scale = 2.0**16 * (replay if replay > 1 and comm.get_world_size() == 1 else 1)
        scaler = (
            torch.amp.GradScaler("cuda", init_scale=init_scale)
            if self.cfg.enable_amp
            else None
        )
        return scaler
