import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class StreamingSegmentorV2(nn.Module):
    """Segmentor for chunk-wise (streaming) backbones such as PT-v3-LaCT-Dec and
    PT-v3-LaCT-Temporal.

    Training: the backbone's `grad_accum_forward` runs forward + backward chunk by chunk
    (gradients accumulate across chunks); the returned loss is only for logging and is
    flagged `_skip_backward` for the trainer.
    Evaluation: one forward over the whole sample, then the linear segmentation head.
    """

    def __init__(self, num_classes, backbone_out_channels, backbone=None, criteria=None):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
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
            feat = point.feat if isinstance(point, Point) else point
            seg_logits = self.seg_head(feat)
            if "segment" in input_dict.keys():
                loss = self.criteria(seg_logits, input_dict["segment"])
                return dict(loss=loss, seg_logits=seg_logits)
            return dict(seg_logits=seg_logits)
