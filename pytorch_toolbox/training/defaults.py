import os
from numbers import Number
from types import SimpleNamespace
from typing import Collection, Optional, Callable, List, Union, Tuple, Dict, Any, Iterator, Iterable

import numpy as np
import torch
from fastprogress.fastprogress import MasterBar, ProgressBar
from torch import nn as nn, optim, Tensor


def num_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(1, num_cpus())
_default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
defaults = SimpleNamespace(device=_default_device, cpus=_default_cpus)
default_collate = torch.utils.data.dataloader.default_collate
default_lr = slice(3e-3)
default_wd = 1e-2

ModuleList = Collection[nn.Module]
NPArray = np.ndarray
OptOptimizer = Optional[optim.Optimizer]
ParamList = Collection[nn.Parameter]
SplitFunc = Callable[[nn.Module], List[nn.Module]]
SplitFuncOrIdxList = Union[Callable, Collection[ModuleList]]
TensorOrNumber = Union[Tensor, Number]
TensorOrNumberList = Collection[TensorOrNumber]
TensorImage = Tensor
TensorImageSize = Tuple[int, int, int]
Tensors = Union[Tensor, Collection['Tensors']]
Weights = Dict[str, Tensor]
HookFunc = Callable[[nn.Module, Tensors, Tensors], Any]
LogitTensorImage = TensorImage
MetricFunc = Callable[[Tensor, Tensor], TensorOrNumber]
MetricFuncList = Collection[MetricFunc]
MetricsList = Collection[TensorOrNumber]
LossFunction = Callable[[Tensor, Tensor], Tensor]
OptionalLossFunction = Optional[LossFunction]
OptionalMetrics = Optional[MetricsList]
OptionalSplitFunction = Optional[SplitFunc]
Floats = Union[float, Collection[float]]
AnnealFunc = Callable[[Number, Number, float], Number]
StartOptEnd = Union[float, Tuple[float, float]]
PBar = Union[MasterBar, ProgressBar]
bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
