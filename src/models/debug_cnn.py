from pytorch_toolbox.utils.training import flatten_model, split_model_idx
from .layers_and_init import *

def debug_cnn(pretrained=None):
    model = nn.Sequential(
        nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
        nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(8, 28)
    )
    n_first_half_layers = len(flatten_model(model[:2]))
    # n_second_half_layers = len(flatten_model(model[3:]))
    layer_groups = split_model_idx(model, [n_first_half_layers])
    model.layer_groups = layer_groups
    return model

