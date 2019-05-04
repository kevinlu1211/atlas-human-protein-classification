from pytorch_toolbox.core import FocalLoss
from pytorch_toolbox.core.losses.focal_loss import calculate_focal_loss


def focal_loss(preds, targs, gamma=2):
    focal_loss = FocalLoss.reshape_to_batch_size_x_minus_one_and_sum_over_last_dimension(
        calculate_focal_loss(preds, targs, gamma=gamma))
    return focal_loss.mean()
