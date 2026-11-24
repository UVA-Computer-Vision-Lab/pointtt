"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch.nn as nn
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight
