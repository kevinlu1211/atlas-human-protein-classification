import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_toolbox.core import BaseLoss
from pytorch_toolbox.core.utils import to_numpy
from pytorch_toolbox.core.defaults import default_hardware


class ArcFaceLoss(BaseLoss):
    def __init__(self, scale=30.0, margin=0.5):
        super().__init__()
        self.classify_loss = nn.CrossEntropyLoss(reduce=False)
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    @property
    def unreduced_loss(self):
        return self._unreduced_loss

    @property
    def per_sample_loss(self):
        return self._per_sample_loss

    @property
    def reduced_loss(self):
        return self._reduced_loss

    def __call__(self, out, *yb):
        """

        :param cosine: this is the output of ArcMarginProduct
        :param labels:
        :return:
        """
        cosine = out
        labels = yb[0]
        loss = calculate_arc_face_loss(cosine, labels, self.cos_m, self.sin_m, self.th, self.mm, self.scale)
        self._unreduced_loss = loss
        self._per_sample_loss = loss
        self._reduced_loss = self._per_sample_loss.mean()
        return self._reduced_loss


def calculate_arc_face_loss(cosine, labels, cos_m, sin_m, th, mm, scale):
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

    # Double angle formula cos(A + B) = cos(A)*cos(B) - sin(A)*sin(B)
    cosine_with_margin = cosine * cos_m - sine * sin_m

    # Only include samples where adding the margin will aid in training, as for all cosine < cosine(pi - margin),
    # adding a margin will decrease the target logit => the loss will be higher.
    # If phi > cosine(pi - margin),
    cosine_with_margin = torch.where(cosine > th, cosine_with_margin, cosine - mm)

    one_hot_labels = labels
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot_labels * cosine_with_margin) + ((1.0 - one_hot_labels) * cosine)
    output *= scale
    labels = torch.from_numpy(np.where(to_numpy(labels) == 1)[1]).to(default_hardware.device)
    loss1 = nn.CrossEntropyLoss(reduce=False)(output, labels)
    loss2 = nn.CrossEntropyLoss(reduce=False)(cosine, labels)
    gamma = 1
    loss = (loss1 + gamma * loss2) / (1 + gamma)
    return loss


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample

            cos(theta) = (A * B) / (||A||*||B||)

        """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine
