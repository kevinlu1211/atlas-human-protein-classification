from pytorch_toolbox.core.losses import FocalLoss
from pytorch_toolbox.core.losses.focal_loss import calculate_focal_loss


def focal_loss(preds, targs, gamma=2):
    focal_loss = FocalLoss.sum_over_last_dimension(calculate_focal_loss(preds, targs, gamma=gamma))
    return focal_loss.mean()