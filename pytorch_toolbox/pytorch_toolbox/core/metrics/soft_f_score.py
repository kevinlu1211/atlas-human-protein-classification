import numpy as np
import torch

from pytorch_toolbox.core import calculate_soft_f_score_loss
from pytorch_toolbox.core.utils import to_numpy


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_f_score(preds, targs, beta, th=0.5, d=50.0):
    if isinstance(preds, torch.Tensor) and isinstance(targs, torch.Tensor):
        return (-1. * calculate_soft_f_score_loss(preds, targs, beta=beta).mean()) + 1
    else:
        preds = to_numpy(preds)
        targs = to_numpy(targs)
        preds = sigmoid_np(d * (preds - th))
        targs = targs.astype(np.float)
        score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
        return score
