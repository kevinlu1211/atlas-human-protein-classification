import re
from enum import Enum

from pytorch_toolbox.core.defaults import *


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().data.numpy()
    else:
        return np.array(t)


def listify(p: OptionalListOrItem = None, q: OptionalListOrItem = None):
    "Make `p` same length as `q`"
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def if_none(a: Any, b: Any) -> Any:
    return b if a is None else a


def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list, np.ndarray))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def str_to_float(x):
    if isinstance(x, str):
        return float(x)
    elif is_listy(x):
        return [str_to_float(s) for s in x]
    else:
        return x


def range_of(x):
    return list(range(len(x)))


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def determine_phase(train, last_target, label_key="label"):
    if train:
        return Phase.TRAIN
    else:
        label = last_target.get(label_key)
        if label is not None:
            return Phase.VAL
        else:
            return Phase.TEST


def camel2snake(name: str) -> str:
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def even_mults(start: float, stop: float, n: int) -> np.ndarray:
    "Build evenly stepped schedule from `start` to `stop` in `n` steps."
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step ** i) for i in range(n)])


def make_one_hot(labels, n_classes=28):
    one_hots = []
    for label in labels:
        one_hot = np.zeros(n_classes)
        for label_idx in label:
            one_hot[label_idx] = 1
        one_hots.append(one_hot.astype(np.float32))
    return one_hots

