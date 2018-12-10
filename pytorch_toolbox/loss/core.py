from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from fastai import *
import fastai


def calculate_ce_loss(preds, targets, weight_maps=None, weight=1):
    """

    :param preds: B x C x H x W
    :param targets: B x 1 x H x W
    :param weight_maps: B x 1 x H x W
    :param weight: 1 x 1
    :return:
    """
    ce_loss_criterion = nn.CrossEntropyLoss(reduce=False)
    if targets.shape[0] == 1:
        targets = targets.squeeze(1)
    else:
        targets = targets.squeeze()
    loss = ce_loss_criterion(preds, targets.long())
    loss *= weight
    if weight_maps is not None:
        loss *= weight_maps.squeeze()
    return loss


def calculate_dice_loss(preds, targets, dice_loss_weights=None, n_classes=2):
    if dice_loss_weights is None:
        dice_loss_weights = {f"weight_for_class_{i}": 1 for i in range(n_classes)}

    dice_loss_ = multi_class_dice_loss(preds, targets.long(), n_classes=n_classes)
    dice_loss_lookup = {}

    for k, v in dice_loss_.items():
        class_label = k[-1]
        dice_loss_lookup[k] = v * dice_loss_weights[f"weight_for_class_{class_label}"]

    total_dice_loss = torch.mean(torch.stack([loss for loss in dice_loss_lookup.values()]))
    return total_dice_loss, dice_loss_lookup


def multi_class_dice_loss(pred_logits, targets, n_classes):
    """

    :param pred_logits: B x C x H x W
    :param targets: B x 1 x H x W, type Long
    :return:
    """

    one_hot = torch.FloatTensor(targets.size(0), n_classes, targets.size(2), targets.size(3)).zero_()
    target_one_hot = one_hot.scatter_(1, targets.cpu().data, 1).to(fastai.defaults.device)
    pred_probs = F.softmax(pred_logits, dim=1)
    batch_size = pred_logits.size(0)
    loss_dict = defaultdict(list)
    for batch_idx in range(batch_size):
        for class_i in range(n_classes):
            loss_for_class_i = dice_loss(pred_probs[batch_idx, class_i, :, :], target_one_hot[batch_idx, class_i, :, :])
            loss_dict[f"loss_for_class_{class_i}"].append(loss_for_class_i)

    return {k: torch.stack(v) for k, v in loss_dict.items()}


def dice_loss(inputs, one_hots):
    inputs_flat = inputs.contiguous().view(-1)
    one_hots_flat = one_hots.contiguous().view(-1)
    intersection = torch.sum(inputs_flat * one_hots_flat)
    return 1 - ((2 * intersection + 1.) / (torch.sum(inputs_flat) + torch.sum(one_hots_flat) + 1.))


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

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
           ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss

    return loss


# Referenced from
def calculate_f1_soft_loss(logits, labels):
    """
    logits: B x N_classes
    labels: B x N_classes
    """
    __small_value = 1e-6
    beta = 1
    probs = F.sigmoid(logits)
    soft_tp_plus_fp = torch.sum(probs, 1) + __small_value
    # note that this is a bit of a misnomer as the labels are usually 1 hot, but for the case that they aren't this
    # would make a bit more sense
    soft_tp_plus_fn = torch.sum(labels, 1) + __small_value
    true_positive = torch.sum(labels * probs, 1)
    precision = true_positive / soft_tp_plus_fp
    recall = true_positive /soft_tp_plus_fn
    f1_soft = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall + __small_value)
    return 1 - f1_soft
