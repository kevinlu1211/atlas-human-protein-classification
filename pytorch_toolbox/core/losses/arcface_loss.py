import math

import torch
import torch.nn as nn
import torch.functional as F


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, scale=30.0, margin=0.5):
        super().__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine, labels):
        """

        :param cosine: this is the output of ArcMarginProduct
        :param labels:
        :return:
        """
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Double angle formula cos(A + B) = cos(A)*cos(B) - sin(A)*sin(B)
        cosine_with_margin = cosine * self.cos_m - sine * self.sin_m

        # Only include samples where adding the margin will aid in training, as for all cosine < cosine(pi - margin),
        # adding a margin will decrease the target logit => the loss will be higher.
        # If phi > cosine(pi - margin),
        cosine_with_margin = torch.where(cosine > self.th, cosine_with_margin, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine_with_margin) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
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
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine
