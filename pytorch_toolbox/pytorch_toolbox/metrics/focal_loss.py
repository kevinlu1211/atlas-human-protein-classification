from pytorch_toolbox.losses import FocalLoss, focal_loss

def focal_loss_metric(preds, targs, gamma=2):
    loss = FocalLoss.reshape_to_batch_size_x_minus_one_and_sum_over_last_dimension(
        focal_loss(preds, targs, gamma=gamma))
    return loss.mean()
