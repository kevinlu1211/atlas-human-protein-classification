from abc import abstractmethod
import torch.nn as nn

from pytorch_toolbox.callbacks import LearnerCallback
from pytorch_toolbox.utils import is_listy
from pytorch_toolbox.utils.training import flatten_model
from pytorch_toolbox.defaults import Tensor, Tensors, Collection, Sequence, HookFunc


class HookCallback(LearnerCallback):
    "Callback that can be used to register hooks on `modules`. Implement the corresponding function in `self.hook`."

    def __init__(self, learn, modules: Sequence[nn.Module] = None, do_remove: bool = True):
        super().__init__(learn)
        self.modules, self.do_remove = modules, do_remove

    @abstractmethod
    def hook_fn(self):
        pass

    def on_train_begin(self, **kwargs):
        "Register the `Hooks` on `self.modules`."
        if not self.modules:
            self.modules = self._all_modules_with_weights()
        self.hooks = Hooks(self.modules, self.hook_fn)

    def _all_modules_with_weights(self):
        return [m for m in flatten_model(self.learn.model) if hasattr(m, 'weight')]

    def on_train_end(self, **kwargs):
        "Remove the `Hooks`."
        if self.do_remove: self.remove()

    def remove(self):
        if getattr(self, 'hooks', None):
            self.hooks.remove()

    def __del__(self):
        self.remove()


class Hooks:
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, modules: Collection[nn.Module], hook_fn: HookFunc,
                 is_forward: bool = True, detach: bool = True):
        self.hooks = [Hook(m, hook_fn, is_forward, detach) for m in modules]

    def __getitem__(self, i: int):
        return self.hooks[i]

    def __len__(self) -> int: return len(self.hooks)

    def __iter__(self): return iter(self.hooks)

    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hook:
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m: nn.Module, hook_fn: HookFunc, is_forward: bool = True, detach: bool = True):
        self.hook_fn, self.detach, self.stored = hook_fn, detach, None
        register_hook_fn = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = register_hook_fn(self.hook_fn_wrapper)
        self.removed = False

    def hook_fn_wrapper(self, module: nn.Module, input: Tensors, output: Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_fn(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def hook_output(module: nn.Module, detach: bool = True, grad: bool = False) -> Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)


def hook_outputs(modules: Collection[nn.Module], detach: bool = True, grad: bool = False) -> Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def _hook_inner(m, i, o):
    return o if isinstance(o, Tensor) else o if is_listy(o) else list(o)
