import pytorch_toolbox.fastai.fastai as fastai
from .layers_and_init import *

def debug_cnn():
    model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256 , kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(256, 28)
    )
    n_first_half_layers = len(fastai.flatten_model(model[:3]))
    # n_second_half_layers = len(fastai.flatten_model(model[3:]))
    layer_groups = fastai.split_model_idx(model, [n_first_half_layers])
    model.layer_groups = layer_groups
    return model

