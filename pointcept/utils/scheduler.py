"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

Modified for PoinTTT: only OneCycleLR is kept; it is built without the `verbose` argument,
which recent PyTorch releases deprecate.
"""

import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )


def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
