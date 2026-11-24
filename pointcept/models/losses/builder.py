"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg):
        self.criteria = [LOSSES.build(cfg=loss_cfg) for loss_cfg in cfg]

    def __call__(self, pred, target):
        loss = 0
        for c in self.criteria:
            if (target == -1).all():
                # All targets ignored — return a zero loss that's still
                # connected to the graph so chunk-wise backward works.
                # A bare torch.tensor(0.0) has no grad_fn → loss.backward()
                # would raise "element 0 ... does not require grad".
                return (pred.float() * 0.0).sum()
            loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
