import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    """Segmentor of the HOI4D stage-1 PTv3 (PT-v3m1 on independent clips), used for training
    only: one forward over the batch, the linear segmentation head and the training loss."""

    def __init__(self, num_classes, backbone_out_channels, backbone, criteria):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        loss = self.criteria(seg_logits, input_dict["segment"])
        return dict(loss=loss)


@MODELS.register_module()
class StreamingSegmentorV2(nn.Module):
    """Segmentor for chunk-wise (streaming) backbones such as PT-v3-LaCT-Dec and
    PT-v3-LaCT-Temporal.

    Training: the backbone's `grad_accum_forward` runs forward + backward chunk by chunk
    (gradients accumulate across chunks); the returned loss is only for logging and is
    flagged `_skip_backward` for the trainer.
    Evaluation: one forward over the whole sample, then the linear segmentation head.
    """

    def __init__(self, num_classes, backbone_out_channels, backbone, criteria):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if self.training:
            # Zero grad before per-window backward calls
            self.zero_grad()
            total_loss = self.backbone.grad_accum_forward(
                input_dict, self.seg_head, self.criteria
            )
            # Return scalar loss with flag to skip backward in training loop
            loss = torch.tensor(total_loss, device="cuda")
            loss._skip_backward = True
            return dict(loss=loss)
        else:
            point = self.backbone(input_dict)
            seg_logits = self.seg_head(point.feat)
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
