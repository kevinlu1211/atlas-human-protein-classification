import numpy as np
import torch

from pytorch_toolbox.utils import to_numpy


def accuracy(preds, targs, th=0.0):
    if isinstance(preds, torch.Tensor) and isinstance(targs, torch.Tensor):
        preds = (preds > th).int()
        targs = targs.int()
        return (preds == targs).float().mean()
    else:
        preds = to_numpy(preds)
        targs = to_numpy(targs)
        preds = (preds > th).astype(np.int32)
        return (preds == targs).mean()
