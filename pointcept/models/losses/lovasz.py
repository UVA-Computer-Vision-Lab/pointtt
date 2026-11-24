"""
Lovasz Loss
refer https://arxiv.org/abs/1705.08790

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

Adapted from segmentation_models.pytorch (losses/lovasz.py), which builds on pytorch-toolbelt
and the LovaszSoftmax reference implementation by Maxim Berman; all MIT License, see
THIRD_PARTY_NOTICES.md. Modified for PoinTTT: only the multi-class Lovasz-softmax loss over the
classes present in the labels is kept, and the per-class loop iterates over Python ints.
"""

import torch
from torch.nn.modules.loss import _Loss

from .builder import LOSSES


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels):
    """Multi-class Lovasz-Softmax loss over the classes present in labels
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    """
    losses = []
    # Iterate over Python ints: indexing with a 0-d CUDA tensor forced a device
    # synchronisation per class.
    for c in labels.unique().tolist():
        fg = (labels == c).type_as(probas)  # foreground for class c
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def mean(values):
    """Mean of a non-empty list."""
    acc = values[0]
    for v in values[1:]:
        acc += v
    if len(values) == 1:
        return acc
    return acc / len(values)


@LOSSES.register_module()
class LovaszLoss(_Loss):
    def __init__(self, mode: str, ignore_index: int = -1, loss_weight: float = 1.0):
        """Multi-class Lovasz-softmax loss for segmentation (mode "multiclass").
        Args:
            ignore_index: Label that indicates ignored points (does not contribute to loss)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C), logits
             - **y_true** - torch.Tensor of shape (N,)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode == "multiclass"
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        y_pred = y_pred.softmax(dim=1)
        valid = y_true != self.ignore_index
        loss = _lovasz_softmax_flat(y_pred[valid], y_true[valid])
        return loss * self.loss_weight
