import pretrainedmodels
from pytorch_toolbox.utils.training import flatten_model, split_model_idx

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
    n_starting_layers = len(flatten_model(model[:3]))
    n_middle_layers = len(flatten_model(model[3:5]))
    layer_groups = split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
    model.layer_groups = layer_groups
    return model

def se_resnext50_32x4d_four_channel_input_two_fc(pretrained=True):
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
        *list(model.children())[:-2],
        *fc_layers
    )
    n_starting_layers = len(flatten_model(model[:3]))
    n_middle_layers = len(flatten_model(model[3:5]))
    layer_groups = split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
    model.layer_groups = layer_groups
    return model
