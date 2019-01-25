from functools import partial

from .cbam import *
from .gapnet import *
from .resnet import *
from .debug_cnn import *
from .senet import *

model_lookup = {
    "resnet18_four_channel_input": resnet18_four_channel_input,
    "resnet34_four_channel_input": resnet34_four_channel_input,
    "resnet34_d_four_channel_input": resnet34_d_four_channel_input,
    "resnet34_four_channel_input_one_fc": resnet34_four_channel_input_one_fc,
    "resnet34_d_four_channel_input_one_fc": resnet34_d_four_channel_input_one_fc,
    "resnet50_four_channel_input": resnet50_four_channel_input,
    "cbam_resnet18_four_channel_input_one_fc": cbam_resnet18_four_channel_input_one_fc,
    "cbam_resnet34_four_channel_input": cbam_resnet34_four_channel_input,
    "cbam_resnet50_four_channel_input": cbam_resnet50_four_channel_input,
    "cbam_resnet50_four_channel_input_one_fc": cbam_resnet50_four_channel_input_one_fc,
    "cbam_resnet101_four_channel_input": cbam_resnet101_four_channel_input,
    "gapnet_resnet34_four_channel_input_backbone": gapnet_resnet34_four_channel_input_backbone,
    "gapnet_resnet34_d_four_channel_input_backbone": gapnet_resnet34_d_four_channel_input_backbone,
    "gapnet2_resnet34_four_channel_input_backbone": gapnet2_resnet34_four_channel_input_backbone,
    "gapnet2_resnet34_d_four_channel_input_backbone": gapnet2_resnet34_d_four_channel_input_backbone,
    "se_resnext50_32x4d_four_channel_input": se_resnext50_32x4d_four_channel_input,
    "se_resnext50_32x4d_four_channel_input_two_fc": se_resnext50_32x4d_four_channel_input_two_fc,
    "debug_cnn": debug_cnn
}


