from typing import Union, Callable, Tuple, List, Any

from torch import optim

from pytorch_toolbox.training.defaults import ModuleList, Floats
from pytorch_toolbox.utils import split_bn_bias, trainable_params, listify, is_tuple


class OptimizerWrapper:
    "Basic wrapper around an optimizer to simplify HP changes."

    def __init__(self, opt: optim.Optimizer, wd: Floats = 0., true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func: Union[type, Callable], lr: Union[float, Tuple, List],
               layer_groups: ModuleList, **kwargs: Any) -> optim.Optimizer:
        "Create an optim.Optimizer from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr = listify(lr, layer_groups)
        return opt

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    # Pytorch optimizer methods
    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for p in pg1['params']: p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']: p.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        self.opt.zero_grad()

    # Hyperparameters as properties
    @property
    def lr(self) -> float:
        "Get learning rate."
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) -> None:
        "Set learning rate."
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self) -> float:
        "Get momentum."
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        "Set momentum."
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        "Get beta (or alpha as makes sense for given optimizer)."
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        "Get weight decay."
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    # Helper functions
    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key: str, val: Any, bn_groups: bool = True) -> Any:
        "Set the values inside the optimizer dictionary at the key."
        if is_tuple(val): val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key: str) -> Union[List[float], Tuple[List[float], List[float]]]:
        "Read a hyperparameter key in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val


