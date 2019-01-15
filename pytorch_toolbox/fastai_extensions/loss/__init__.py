from .core import FocalLoss, SoftF1Loss, LossWrapper
from .core import *
loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": SoftF1Loss
}
