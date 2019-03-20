import re
from enum import Enum

from pytorch_toolbox.core.defaults import *


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


# def listify(x):
#     if isinstance(x, str):
#         return [x]
#     elif not isinstance(x, Iterable):
#         return [x]
#     elif not isinstance(x, List):
#         return [x]
#     else:
#         return x

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
    return isinstance(x, (tuple, list))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def to_device(t: Tensors, device: torch.device):
    device = if_none(device, defaults.device)
    if is_listy(t):
        return [to_device(o, device) for o in t]
    return t.to(device)


def range_of(x):
    return list(range(len(x)))


def flatten_model(m: nn.Module):
    return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def num_children(m: nn.Module) -> int:
    return len(children(m))


def children(m: nn.Module) -> ModuleList:
    return list(m.children())


def split_bn_bias(layer_groups: ModuleList) -> ModuleList:
    "Sort each layer in  `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups


def trainable_params(m: nn.Module) -> ParamList:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def to_detach(b: Tensors):
    "Recursively detach lists of tensors in `b `"
    if is_listy(b): return [to_detach(o) for o in b]
    return b.detach() if isinstance(b, Tensor) else b


def camel2snake(name: str) -> str:
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def requires_grad(m: nn.Module, b: Optional[bool] = None) -> Optional[bool]:
    "If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param."
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad = b
