from torch.nn import functional as F

from pytorch_toolbox.core.losses import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, gamma=2):
        self.gamma = gamma

    @property
    def unreduced_loss(self):
        return self._unreduced_loss

    @property
    def per_sample_loss(self):
        return self._per_sample_loss

    @property
    def reduced_loss(self):
        return self._reduced_loss

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        # This returns B x ... (same shape as input)
        self._unreduced_loss = calculate_focal_loss(prediction, target, self.gamma)
        self._per_sample_loss = self.reshape_to_batch_size_x_minus_one_and_sum_over_last_dimension(
            self._unreduced_loss)
        self._reduced_loss = self._per_sample_loss.mean()
        return self._reduced_loss


def calculate_focal_loss(input, target, gamma=2):
    """

    :param input: B x N_classes
    :param target: B x N_classes
    :param gamma: the higher the value, the greater the loss for uncertain classes
    :return:
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})"
                         .format(target.size(), input.size()))

    target = target.float()
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
           ((-max_val).exp() + (-input - max_val).exp()).log()

    inv_probs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (inv_probs * gamma).exp() * loss
    return loss
