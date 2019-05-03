from abc import abstractmethod

import torch.nn as nn

from core.utils import camel2snake


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
    def reshape_to_batch_size_x_minus_one_and_sum_over_last_dimension(tensor):
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


