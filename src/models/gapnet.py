import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_toolbox.fastai.fastai as fastai


class GapNet(nn.Module):

    def __init__(self):
        super().__init__()
        conv_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2)
        selu_1 = nn.SELU()
        pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_1 = nn.Sequential(conv_1, selu_1, pool_1)
        self.gap_1 = nn.AdaptiveAvgPool2d(1)
        conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        selu_2 = nn.SELU()
        conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        selu_2_1 = nn.SELU()
        conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        selu_2_2= nn.SELU()
        conv_2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        selu_2_3 = nn.SELU()
        self.block_2 = nn.Sequential(conv_2, selu_2,
                                     conv_2_1, selu_2_1,
                                     conv_2_2, selu_2_2,
                                     conv_2_3, selu_2_3)
        self.gap_2 = nn.AdaptiveAvgPool2d(1)
        conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        selu_3_1 = nn.SELU()
        conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        selu_3_2 = nn.SELU()
        conv_3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        selu_3_3 = nn.SELU()
        self.block_3 = nn.Sequential(conv_3_1, selu_3_1,
                                     conv_3_2, selu_3_2,
                                     conv_3_3, selu_3_3)
        self.gap_3 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(224, 256), nn.SELU())
        self.fc2 = nn.Sequential(nn.Linear(256, 28))


    def forward(self, x):
        bs = x.size(0)
        x = self.block_1(x)
        gap_1 = self.gap_1(x).view(bs, -1)
        x = self.block_2(x)
        gap_2 = self.gap_2(x).view(bs, -1)
        x = self.block_3(x)
        gap_3 = self.gap_3(x).view(bs, -1)
        concat = torch.cat([gap_1, gap_2, gap_3], dim=-1)
        out = self.fc1(concat)
        out = self.fc2(out)
        return out


class TransparentGAP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.feature_maps = F.adaptive_avg_pool2d(x, 1)
        return x

class DynamicGapNet(nn.Module):
    def __init__(self, encoder, n_classes, gap_layer_idxs=None, dropout_prob=0.25):
        """
        encoder: nn.Sequential object
        gap_feature_start_idx: these are the layers after which a GAP layer will be placed
        """
        super().__init__()
        sizes, *_ = fastai.callbacks.model_sizes(encoder)
        if gap_layer_idxs is None:
            gap_layer_idxs = range(len(encoder))
        layers = []
        total_n_features = 0
        self.gaps = []
        for i, (size, module) in enumerate(zip(sizes, encoder)):
            layers.append(module)
            if i in gap_layer_idxs:
                total_n_features += size[1]
                gap = TransparentGAP()
                self.gaps.append(gap)
                layers.append(gap)
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(total_n_features, n_classes)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        _ = self.model(x)
        features = torch.cat([gap.feature_maps for gap in self.gaps], dim=1)
        bs = features.size()[0]
        features = features.squeeze()
        if bs == 1:
            features = features.unsqueeze(0)
        features = F.dropout(features, p=self.dropout_prob)
        logits = self.fc(features)
        return logits

