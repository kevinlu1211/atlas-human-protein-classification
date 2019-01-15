from .core import FocalLoss, SoftF1Loss, LossWrapper
loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": SoftF1Loss
}
