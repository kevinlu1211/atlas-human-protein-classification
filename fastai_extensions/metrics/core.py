import torch
import numpy as np
from ...utils import to_numpy
from ..loss import calculate_f1_soft_loss

# Referenced from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def accuracy(preds,targs,th=0.0):
    if isinstance(preds, torch.Tensor) and isinstance(targs, torch.Tensor):
        preds = (preds > th).int()
        targs = targs.int()
        return (preds==targs).float().mean()
    else:
        preds = to_numpy(preds)
        targs = to_numpy(targs)
        preds = (preds > th).astype(np.int32)
        return (preds==targs).mean()

def f1_soft(preds,targs,th=0.5,d=50.0):
    if isinstance(preds, torch.Tensor) and isinstance(targs, torch.Tensor):
        return (-1. * calculate_f1_soft_loss(preds, targs).mean()) + 1
    else:
        preds = to_numpy(preds)
        targs = to_numpy(targs)
        preds = sigmoid_np(d*(preds - th))
        targs = targs.astype(np.float)
        score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
        return score

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": f1_soft
}