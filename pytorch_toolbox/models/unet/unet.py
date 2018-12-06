import torch.nn as nn
import torchvision
from fastai import *
import fastai


def create_encoder(name, pretrained=True):
    if name == "pytorch_resnet18":
        return nn.Sequential(*list(torchvision.models.resnet.resnet18(pretrained=pretrained).children())[:-2])
    elif name == "pytorch_resnet34":
        return nn.Sequential(*list(torchvision.models.resnet.resnet34(pretrained=pretrained).children())[:-2])
    elif name == "pytorch_resnet50":
        return nn.Sequential(*list(torchvision.models.resnet.resnet50(pretrained=pretrained).children())[:-2])
    elif name == "pytorch_resnet101":
        return nn.Sequential(*list(torchvision.models.resnet.resnet101(pretrained=pretrained).children())[:-2])
    elif name == "pytorch_resnet152":
        return nn.Sequential(*list(torchvision.models.resnet.resnet152(pretrained=pretrained).children())[:-2])
    else:
        print(f"A method to extract the encoder from {name} is not defined")


def dynamic_unet(encoder, n_classes=1, pretrained=True):
    encoder = create_encoder(encoder, pretrained=pretrained)
    unet = fastai.vision.models.DynamicUnet(encoder, n_classes=n_classes)
    return encoder, unet