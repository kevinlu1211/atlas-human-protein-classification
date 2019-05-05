from pathlib import Path
from typing import Callable, Collection, Union, Optional, Iterator, Tuple
from functools import partial

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from fastprogress import progress_bar, master_bar
from miniutils import progbar

from pytorch_toolbox.training.learner.train import fit_one_cycle, fit_multi_step, lr_find, to_fp16
from pytorch_toolbox.callbacks import Callback, CallbackList, CallbackHandler, Recorder
from pytorch_toolbox.data import DataBunch
from pytorch_toolbox.defaults import Floats, default_lr, bn_types, LossFunction, OptionalMetrics, \
    OptionalLossFunction, PBar, OptionalOptimizer
from pytorch_toolbox.training import OptimizerWrapper
from pytorch_toolbox.utils import listify, if_none, is_listy, to_numpy, even_mults, Phase
from pytorch_toolbox.utils.training import flatten_model, to_detach, requires_grad

AdamW = partial(Adam, betas=(0.9, 0.99))


## TODO: refactor fit, validate, and predict_on_dl as they share a lot of common functionality

class Learner:
    def __init__(self, data: DataBunch, model: nn.Module, loss_func: Callable, opt_func: Callable = AdamW,
                 metrics: Collection[Callable] = None, true_weight_decay: bool = True,
                 batch_norm_weight_decay: bool = True, weight_decay: Floats = 1e-2, train_bn: bool = True,
                 path: str = ".", model_dir: str = "model", callback_fns: Collection[Callable] = None,
                 callbacks: Collection[Callable] = [], layer_groups: Collection[nn.Module] = None):
        """

        :param data: object that wraps the data loaders
        :param model: the model
        :param loss_func: the loss function
        :param opt_func: the optimizer
        :param metrics: metrics to be calculated after each epoch
        :param true_weight_decay:
        :param batch_norm_weight_decay:
        :param weight_decay: the weight decay
        :param train_bn:
        :param path: the root path of the Learner
        :param model_dir: the name of the directory that the model will be saved in, where path is the parent
        :param callback_fns: the learner callback functions
        :param callbacks: the callback function
        :param layer_groups: the different layers that the model is split into for differential learning rates
        """
        self.data = data
        self.model = model.to(self.data.device)
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.true_weight_decay = true_weight_decay
        self.batch_norm_weight_decay = batch_norm_weight_decay
        self.weight_decay = weight_decay
        self.train_bn = train_bn
        self.path = Path(path)
        self.model_dir = model_dir
        self.metrics = listify(metrics)
        self.callbacks = listify(callbacks)
        self.callback_fns = [Recorder] + listify(callback_fns)

        if layer_groups is None:
            self.layer_groups = [nn.Sequential(*flatten_model(self.model))]
        else:
            self.layer_groups = layer_groups

    def fit(self, epochs: int, lr: Union[Floats, slice] = default_lr,
            wd: Floats = None, callbacks: Collection[Callback] = None) -> None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None:
            wd = self.weight_decay
        self.create_opt(lr, wd)
        callbacks = [cb(self) for cb in self.callback_fns] + listify(callbacks)
        fit(epochs, self.model, self.loss_func, opt=self.opt, data=self.data, metrics=self.metrics,
            callbacks=self.callbacks + callbacks)

    def create_opt(self, lr: Floats, wd: Floats = 0.) -> None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = OptimizerWrapper.create(opt_func=self.opt_func, lr=lr, layer_groups=self.layer_groups,
                                           wd=wd, true_wd=self.true_weight_decay, bn_wd=self.batch_norm_weight_decay)

    def lr_range(self, lr: Union[float, slice]) -> np.ndarray:
        "Build differential learning rates."
        if not isinstance(lr, slice):
            return lr
        if lr.start:
            res = even_mults(lr.start, lr.stop, len(self.layer_groups))
        else:
            res = [lr.stop / 3] * (len(self.layer_groups) - 1) + [lr.stop]
        return np.array(res)

    def model_gradients(self):
        for lg in self.layer_groups:
            for l in lg:
                print(l)
                for p in l.parameters():
                    print(p.shape)
                    print(p.requires_grad)

    def _predict_on_dl(self, dl, phase, callbacks=None, callback_fns=None, metrics=None):
        assert dl is not None, "A dataloader must be provided"
        assert phase is not None, "A phase must be provided: {Phase.TRAIN, Phase.VAL, Phase.TEST}"
        callbacks_fns = [cb(self) for cb in if_none(callback_fns, [])]
        callbacks = self.callbacks + if_none(callbacks, []) + if_none(callbacks_fns, [])

        if phase is not Phase.TEST:
            metrics = if_none(metrics, self.metrics)
        else:
            metrics = []
        cb_handler = CallbackHandler(callbacks=callbacks,
                                     metrics=metrics)
        with torch.no_grad():
            self.model.eval()
            cb_handler.on_epoch_begin()
            for xb, yb in progbar(dl):
                xb, yb = cb_handler.on_batch_begin(xb, yb, train=False, phase=phase)
                if not is_listy(xb):
                    xb = [xb]
                out = self.model(*xb)
                out = cb_handler.on_loss_begin(out)
                out = cb_handler.on_batch_end(out)

    def predict_on_train_dl(self, callbacks=None, callback_fns=None, metrics=None):
        dl = self.data.train_dl
        self._predict_on_dl(dl, Phase.TRAIN, callbacks, callback_fns, metrics)

    def predict_on_val_dl(self, callbacks=None, callback_fns=None, metrics=None):
        """Test with callbacks"""
        dl = self.data.valid_dl
        self._predict_on_dl(dl, Phase.VAL, callbacks, callback_fns, metrics)

    def predict_on_test_dl(self, callbacks=None, callback_fns=None, metrics=None):
        """Test with callbacks"""
        dl = self.data.test_dl
        self._predict_on_dl(dl, Phase.TEST, callbacks, callback_fns, metrics)

    def freeze_to(self, n: int) -> None:
        "Freeze layers up to layer `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self) -> None:
        "Freeze up to last layer."
        assert (len(self.layer_groups) > 1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def freeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        super().unfreeze()
        for i in layer_group_idxs:
            for l in self.layer_groups[i]:
                if not self.train_bn or not isinstance(l, bn_types):
                    requires_grad(l, False)

    def unfreeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        layer_group_idxs_to_freeze = list(set(list(range(len(self.layer_groups)))) - set(layer_group_idxs))
        self.freeze_layer_groups(layer_group_idxs_to_freeze)

    def load_model_with_name(self, name, device=None):
        if device is None:
            device = self.data.device
        self.model.load_state_dict(torch.load(self.path / self.model_dir / f"{name}.pth", map_location=device))
        return self

    def load_model_with_path(self, path, device=None):
        if device is None:
            device = self.data.device
        self.model.load_state_dict(torch.load(path, map_location=device))
        return self

    def save_model_with_name(self, name, return_path: bool = False) -> Union[None, str]:
        "Save model with `name` to `self.model_dir`, and return path if `return_path`."
        path = Path(self.path) / self.model_dir / f"{name}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        if return_path:
            return path

    def save_model_with_path(self, path, return_path: bool = False) -> Union[None, str]:
        "Save model with `name` to `self.model_dir`, and return path if `return_path`."

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        if return_path:
            return path


Learner.fit_one_cycle = fit_one_cycle
Learner.fit_multi_step = fit_multi_step
Learner.lr_find = lr_find
Learner.to_fp16 = to_fp16


def fit(epochs: int, model: nn.Module, loss_func: LossFunction, opt: optim.Optimizer,
        data: DataBunch, callbacks: Optional[CallbackList] = None, metrics: OptionalMetrics = None) -> None:
    "Fit the `model` on `data` and learn using `loss` and `opt`."
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception = False
    try:
        for epoch in pbar:
            model.train()
            cb_handler.on_epoch_begin()

            for xb, yb in progress_bar(data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb, phase=Phase.TRAIN)
                loss = loss_batch(model, xb, yb, loss_func, opt, cb_handler)
                if cb_handler.on_batch_end(loss):
                    break

            if hasattr(data, 'valid_dl') and data.valid_dl is not None:
                val_loss = validate(model, data.valid_dl, loss_func=loss_func,
                                    cb_handler=cb_handler, pbar=pbar)
            else:
                val_loss = None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise e
    finally:
        cb_handler.on_train_end(exception)


def loss_batch(model: nn.Module, xb: Tensor, yb: Tensor, loss_func: OptionalLossFunction = None,
               opt: OptionalOptimizer = None,
               cb_handler: Optional[CallbackHandler] = None) -> Tuple[Union[Tensor, int, float, str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = if_none(cb_handler, CallbackHandler())
    if not is_listy(xb):
        xb = [xb]
    if not is_listy(yb):
        yb = [yb]
    out = model(*xb)
    out = cb_handler.on_loss_begin(out)

    if not loss_func:
        return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return loss.detach().cpu()


def validate(model: nn.Module, dl: DataLoader, loss_func: OptionalLossFunction = None,
             cb_handler: Optional[CallbackHandler] = None,
             pbar: Optional[PBar] = None, average=True, n_batch: Optional[int] = None) -> Iterator[
    Tuple[Union[Tensor, int], ...]]:
    "Calculate loss and metrics for the validation set."
    model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler:
                xb, yb = cb_handler.on_batch_begin(xb, yb, train=False, phase=Phase.VAL)
            val_losses.append(loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler))
            if not is_listy(yb):
                yb = [yb]
            nums.append(yb[0].shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]):
                break
            if n_batch and (len(nums) >= n_batch):
                break
        nums = np.array(nums, dtype=np.float32)
        if average:
            return (to_numpy(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:
            return val_losses
