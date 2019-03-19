from .core import BaseLoss, LovaszHingeFlatLoss, LossWrapper
from pytorch_toolbox.core.loss.soft_f1_loss import SoftF1Loss
from pytorch_toolbox.core.loss.focal_loss import FocalLoss

loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": SoftF1Loss,
    "LovaszHingeFlatLoss": LovaszHingeFlatLoss
}
