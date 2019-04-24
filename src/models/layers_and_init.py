import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


def kaiming_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except AttributeError: # if the bias doesn't exist
            pass