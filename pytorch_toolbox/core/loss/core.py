from .lovasz_loss import lovasz_hinge_flat
from fastai import *

from abc import abstractmethod

import torch.nn as nn

from pytorch_toolbox.core.utils import camel2snake


class LossWrapper(nn.Module):
    def __init__(self, losses):
        super().__init__()
        for loss in losses:
            setattr(self, camel2snake(loss.__class__.__name__), loss)
        self.losses = losses

    def forward(self, out, *yb):
        total_loss = sum([l(out, *yb) for l in self.losses])
        return total_loss


class BaseLoss:

    @staticmethod
    def reshape_to_batch_x_minus_one_and_sum_over_last_dimension(tensor):
        batch_size = tensor.size(0)
        return tensor.view(batch_size, -1).sum(dim=1)

    @property
    @abstractmethod
    def unreduced_loss(self):
        pass

    @property
    @abstractmethod
    def per_sample_loss(self):
        pass

    @property
    @abstractmethod
    def reduced_loss(self):
        pass


class LovaszHingeFlatLoss:
    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        original_shape = target.shape
        lovasz_loss = lovasz_hinge_flat(prediction.flatten(), target.flatten(), reduce=False)
        self.per_sample_loss = lovasz_loss.view(*original_shape).sum(dim=1)
        self.loss = lovasz_loss.view(*original_shape)
        return lovasz_loss.view(*original_shape).sum(dim=1).mean()


