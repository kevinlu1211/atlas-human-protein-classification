from pytorch_toolbox.core.training.utils import flatten_model, split_model_idx
from .layers_and_init import *

def debug_cnn():
    model = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(10, 28)
    )
    n_first_half_layers = len(flatten_model(model[:3]))
    # n_second_half_layers = len(fastai.flatten_model(model[3:]))
    layer_groups = split_model_idx(model, [n_first_half_layers])
    model.layer_groups = layer_groups
    return model

