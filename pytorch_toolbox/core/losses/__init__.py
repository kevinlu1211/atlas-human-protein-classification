from functools import partial
from .core import BaseLoss, LossWrapper
from pytorch_toolbox.core.losses.lovasz_loss import LovaszHingeFlatLoss
from pytorch_toolbox.core.losses.soft_f_score_loss import SoftFScoreLoss
from pytorch_toolbox.core.losses.focal_loss import FocalLoss

loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": partial(SoftFScoreLoss, beta=1),
    "SoftF2Loss": partial(SoftFScoreLoss, beta=2),
    "SoftFLoss": SoftFScoreLoss,
    "LovaszHingeFlatLoss": LovaszHingeFlatLoss
}
