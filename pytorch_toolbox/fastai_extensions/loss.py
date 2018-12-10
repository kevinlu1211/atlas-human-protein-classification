import torch
import torch.nn as nn

from fastai import *
import fastai
from ..loss import calculate_ce_loss, calculate_dice_loss, calculate_focal_loss, calculate_f1_soft_loss


class SoftDiceLoss:
    def __init__(self, dice_loss_weights=None, n_classes=2):
        assert n_classes > 1, "Even if it is a binary classification, please use 2 classes instead of one"
        self.dice_loss_weights = dice_loss_weights
        self.n_classes = n_classes

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        loss, individual_losses = calculate_dice_loss(prediction, target, self.dice_loss_weights, self.n_classes)
        self.loss = individual_losses
        return loss


class CELoss:
    def __init__(self, weight_map_key=""):
        self.weight_map_key = weight_map_key

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        #         weight_map = yb[1]
        ce_loss = calculate_ce_loss(prediction, target)
        self.loss = ce_loss
        return torch.mean(ce_loss)


class FocalLoss:
    def __init__(self, gamma=2):
        self.gamma = gamma

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        # This returns B x ... (same shape as input)
        focal_loss = calculate_focal_loss(prediction, target, self.gamma).sum(dim=1)
        self.loss = focal_loss
        return focal_loss.mean()


class SoftF1Loss:
    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]

        # This returns B x 1
        f1_soft_loss = calculate_f1_soft_loss(prediction, target)
        self.loss = f1_soft_loss
        return f1_soft_loss.mean()


class LossWrapper(nn.Module):
    def __init__(self, losses):
        super().__init__()
        for loss in losses:
            setattr(self, fastai.camel2snake(loss.__class__.__name__), loss)
        self.losses = losses

    def forward(self, out, *yb):
        total_loss = sum([l(out, *yb) for l in self.losses])
        return total_loss

