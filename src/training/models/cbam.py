from collections import OrderedDict

from pytorch_toolbox.vision.models import cbam
from .layers_and_init import *

def cbam_resnet18_four_channel_input_one_fc():
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = cbam.cbam_resnet34(num_classes=1000)
    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Linear(in_features=512, out_features=28, bias=True)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def cbam_resnet34_four_channel_input():
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = cbam.cbam_resnet34(num_classes=1000)
    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=256, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=28, bias=True)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def cbam_resnet50_four_channel_input(pretrained=True, checkpoint_path="../model_checkpoints/RESNET50_CBAM.pth"):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = cbam.cbam_resnet50(num_classes=1000)

    if pretrained:
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Get rid of the module. prefix
        updated_checkpoint_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            updated_checkpoint_state_dict[k[7:]] = v
        model.load_state_dict(updated_checkpoint_state_dict)
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


def cbam_resnet50_four_channel_input_one_fc(pretrained=True, checkpoint_path="../model_checkpoints/RESNET50_CBAM.pth"):
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = cbam.cbam_resnet50(num_classes=1000)

    if pretrained:
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Get rid of the module. prefix
        updated_checkpoint_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            updated_checkpoint_state_dict[k[7:]] = v
        model.load_state_dict(updated_checkpoint_state_dict)
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=28, bias=True),
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
        *fc_layers
    )

    return model


def cbam_resnet101_four_channel_input():
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = cbam.cbam_resnet101(num_classes=28)
    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=512, bias=True),
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