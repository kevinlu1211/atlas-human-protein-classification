from functools import partial
from .core import BaseLoss, LossWrapper
from .arcface_loss import ArcFaceLoss
from .lovasz_loss import LovaszHingeFlatLoss
from .soft_f_score_loss import SoftFScoreLoss
from .focal_loss import FocalLoss

loss_lookup = {
    "ArcFaceLoss": ArcFaceLoss,
    "FocalLoss": FocalLoss,
    "CrossEntropyLoss": partial(FocalLoss, gamma=0),
    "SoftFLoss": SoftFScoreLoss,
    "SoftF1Loss": partial(SoftFScoreLoss, beta=1),
    "SoftF2Loss": partial(SoftFScoreLoss, beta=2),
    "LovaszHingeFlatLoss": LovaszHingeFlatLoss
}
