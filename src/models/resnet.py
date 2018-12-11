from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.models import cbam
from .layers_and_init import *


def resnet18_four_channel_input(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet18(num_classes=1000, pretrained=True)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=28, bias=True)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def resnet34_four_channel_input(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet34(num_classes=1000, pretrained=True)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=28, bias=True)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def resnet34_four_channel_input_one_fc(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet34(num_classes=1000, pretrained=True)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Linear(in_features=1024, out_features=28, bias=True),
    )

    fc_layers.apply(kaiming_init)


    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        fc_layers
    )

    return model


def resnet50_four_channel_input(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet50(num_classes=1000, pretrained=pretrained)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=1024, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=28, bias=True)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def resnet50_four_channel_input_one_fc(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet50(num_classes=1000, pretrained=pretrained)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=28, bias=True),
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def resnet152_four_channel_input(pretrained=True):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet152(num_classes=1000, pretrained=True)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-1],
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=512, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=28, bias=True)
    )

    return model