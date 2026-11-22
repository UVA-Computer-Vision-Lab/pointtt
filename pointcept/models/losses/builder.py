"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry
import torch

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            try:
                if (target == -1).all():
                    # All targets ignored — return a zero loss that's still
                    # connected to the graph so chunk-wise backward works.
                    # A bare torch.tensor(0.0) has no grad_fn → loss.backward()
                    # would raise "element 0 ... does not require grad".
                    return (pred.float() * 0.0).sum()
                loss += c(pred, target)
            except Exception as e:
                print(f"Error computing loss: {e}")
                print(
                    f"Shape mismatch detected: pred.shape={pred.shape}, target.shape={target.shape}"
                )
                raise
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
