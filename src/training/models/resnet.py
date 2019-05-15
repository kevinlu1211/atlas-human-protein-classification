import torchvision

from pytorch_toolbox.utils.training import flatten_model, split_model_idx
from pytorch_toolbox.losses.arcface_loss import ArcMarginProduct
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def resnet34_d_four_channel_input(pretrained=False):
    from torchvision.models.resnet import ResNet, BasicBlock
    class ResNet_D(ResNet):
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
    if pretrained:
        raise ValueError("Can't use pretrained model")
    model = ResNet_D(BasicBlock, [3, 4, 6, 3])
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(first_layer_conv.weight, mode='fan_out', nonlinearity='relu')


    model = nn.Sequential(
        first_layer_conv,
        *list(model.children())[1:-2],
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

def resnet34_d_four_channel_input_one_fc(pretrained=False):
    from torchvision.models.resnet import ResNet, BasicBlock
    class ResNet_D(ResNet):
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

    if pretrained:
        raise ValueError("Can't use pretrained model")
    model = ResNet_D(BasicBlock, [3, 4, 6, 3])
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(first_layer_conv.weight, mode='fan_out', nonlinearity='relu')

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

def resnet34_d_four_channel_input_two_fc(pretrained=False):
    from torchvision.models.resnet import ResNet, BasicBlock
    class ResNet_D(ResNet):
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

    if pretrained:
        raise ValueError("Can't use pretrained model")
    model = ResNet_D(BasicBlock, [3, 4, 6, 3])
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(first_layer_conv.weight, mode='fan_out', nonlinearity='relu')

    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(),
        nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=1024, out_features=28, bias=True)
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
    n_starting_layers = len(flatten_model(model[:6]))
    n_middle_layers = len(flatten_model(model[6:8]))
    model.layer_groups = split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])

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

def resnet50_four_channel_input_arc_margin_product(pretrained=True, arc_margin_product_n_out=None):
    assert arc_margin_product_n_out is not None
    first_layer_conv = nn.Conv2d(4, 64, kernel_size=7, stride=3, padding=3, bias=False)
    model = torchvision.models.resnet50(num_classes=1000, pretrained=pretrained)

    if pretrained:
        pretrained_conv_weights = list(model.children())[0].weight
        first_layer_conv.weight = torch.nn.Parameter(
            torch.cat((pretrained_conv_weights, pretrained_conv_weights[:, :1, :, :]), dim=1))

    arc_margin_product = ArcMarginProduct(4096, arc_margin_product_n_out)
    fc_layers = nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5),
        arc_margin_product
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