import pretrainedmodels
import torch
import torch.nn as nn

import pytorch_toolbox.fastai.fastai as fastai
from .layers_and_init import *



def se_resnext50_32x4d_four_channel_input(pretrained=True):
    if pretrained:
        model = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')

    else:
        model = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained=None)

    first_layer_conv_weights = list(model.children())[0][0].weight
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    first_layer_conv.weight = torch.nn.Parameter(
        torch.cat((first_layer_conv_weights, first_layer_conv_weights[:, :1, :, :]), dim=1))
    list(model.children())[0][0] = first_layer_conv

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 28, bias=True),
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        *list(model.children())[:-2],
        *fc_layers
    )
    n_starting_layers = len(fastai.flatten_model(model[:3]))
    n_middle_layers = len(fastai.flatten_model(model[3:5]))
    layer_groups = fastai.split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
    model.layer_groups = layer_groups
    return model
