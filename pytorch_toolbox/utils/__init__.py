import collections

import numpy as np
import torch


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().data.numpy()
    else:
        return np.array(t)


def num_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def make_one_hot(labels, n_classes=28):
    one_hots = []
    for label in labels:
        one_hot = np.zeros(n_classes)
        for label_idx in label:
            one_hot[label_idx] = 1
        one_hots.append(one_hot.astype(np.float32))
    return one_hots


def listify(x):
    if isinstance(x, str):
        return [x]
    elif not isinstance(x, collections.Iterable):
        return [x]
    elif not isinstance(x, collections.List):
        return [x]
    else:
        return x