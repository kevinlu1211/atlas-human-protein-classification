import torch.nn as nn
from .layers_and_init import *

def debug_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(10, 28)
    )
    return model