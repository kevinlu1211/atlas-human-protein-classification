import numpy as np
import torch

def to_numpy(t):
    if isinstance(t, torch.Tensor): return t.cpu().data.numpy()
    else: return np.array(t)



