#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
logger = logging.get_logger("cadlgi")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()
        self.label_smooth = cfg.SOLVER.LABEL_SMOOTH if cfg is not None else 0.0

    def loss(self, logits, targets, per_cls_weights, kwargs):
        # normalize logits shape: allow (B,C) or (B,T,C) -> (B,C)
        if logits.dim() == 3:
            # average over token/time dimension
            logits = logits.mean(dim=1)
        elif logits.dim() == 1:
            # expand to (B,1) to avoid shape errors; useful for binary with single logit
            logits = logits.unsqueeze(1)

        # normalize targets: allow class indices (B) or one-hot (B,C)
        if targets.dim() == 2 and targets.size(1) == logits.size(1):
            targets = targets.argmax(dim=1)

        weight = torch.tensor(per_cls_weights, device=logits.device)
        loss = F.cross_entropy(logits, targets, weight, reduction="none",
                               label_smoothing=self.label_smooth)
        return torch.sum(loss) / targets.shape[0]


LOSS = {
    "softmax": SoftmaxLoss,
}


def build_loss(cfg):
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)
