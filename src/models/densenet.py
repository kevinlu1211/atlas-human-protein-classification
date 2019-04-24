import pretrainedmodels
from .layers_and_init import *

def densenet121_four_channel_input_two_fc(pretrained=True, fc1_dropout=0.5, fc1_bias=True, fc1_out=1024, fc2_dropout=0.5, fc2_bias=True, fc2_out=28):
    if pretrained:
        model = pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet')

    else:
        model = pretrainedmodels.densenet121(num_classes=1000, pretrained=None)

    first_layer_conv_weights = list(model.children())[0][0].weight
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    first_layer_conv.weight = torch.nn.Parameter(
        torch.cat((first_layer_conv_weights, first_layer_conv_weights[:, :1, :, :]), dim=1))
    list(model.children())[0][0] = first_layer_conv

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=fc1_dropout),
        nn.Linear(in_features=2048, out_features=fc1_out, bias=fc1_bias),
        nn.Sigmoid(),
        nn.BatchNorm1d(fc1_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=fc2_dropout),
        nn.Linear(in_features=fc1_out, out_features=fc2_out, bias=fc2_bias)
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        *list(model.children())[0],
        *fc_layers
    )
    return model

def densenet121_four_channel_input_one_fc(pretrained=True, fc1_dropout=0.5, fc1_bias=True, fc1_out=28):
    if pretrained:
        model = pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet')

    else:
        model = pretrainedmodels.densenet121(num_classes=1000, pretrained=None)

    first_layer_conv_weights = list(model.children())[0][0].weight
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    first_layer_conv.weight = torch.nn.Parameter(
        torch.cat((first_layer_conv_weights, first_layer_conv_weights[:, :1, :, :]), dim=1))
    list(model.children())[0][0] = first_layer_conv

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Tanh(),
        nn.Dropout(p=fc1_dropout),
        nn.Linear(in_features=2048, out_features=fc1_out, bias=fc1_bias),
    )

    fc_layers.apply(kaiming_init)

    model = nn.Sequential(
        *list(model.children())[0],
        *fc_layers
    )
    return model



