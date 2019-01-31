from .core import FocalLoss, SoftF1Loss, LovaszHingeFlatLoss, LossWrapper
loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": SoftF1Loss,
    "LovaszHingeFlatLoss": LovaszHingeFlatLoss
}
