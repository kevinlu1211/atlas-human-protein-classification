from pytorch_toolbox.core.callbacks import CallbackList, LRFinder, OneCycleScheduler, MixedPrecision
from pytorch_toolbox.core.defaults import *
from pytorch_toolbox.core.utils import if_none, is_listy
from pytorch_toolbox.core.training.utils import model2half


def fit_one_cycle(learn, cyc_len: int, max_lr: Union[Floats, slice] = default_lr,
                  moms: Tuple[float, float] = (0.95, 0.85), div_factor: float = 25., pct_start: float = 0.3,
                  wd: float = None, callbacks: Optional[CallbackList] = None, **kwargs) -> None:
    "Fit a model following the 1cycle policy."
    max_lr = learn.lr_range(max_lr)
    callbacks = if_none(callbacks, [])
    callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms, div_factor=div_factor,
                                       pct_start=pct_start, **kwargs))
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)


def lr_find(learn, start_lr: Floats = 1e-7, end_lr: Floats = 10, num_it: int = 100, stop_div: bool = True,
            **kwargs: Any):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss explodes."
    start_lr = np.array(start_lr) if is_listy(start_lr) else start_lr
    end_lr = np.array(end_lr) if is_listy(end_lr) else end_lr
    cb = LRFinder(learn, start_lr, end_lr, num_it, stop_div)
    a = int(np.ceil(num_it / len(learn.data.train_dl)))
    learn.fit(a, start_lr, callbacks=[cb], **kwargs)


def to_fp16(learn, loss_scale: float = 512., flat_master: bool = False):
    "Transform `learn` in FP16 precision."
    learn.model = model2half(learn.model)
    learn.mp_cb = MixedPrecision(learn, loss_scale=loss_scale, flat_master=flat_master)
    learn.callbacks.append(learn.mp_cb)
    return learn


# def mixup(learn, alpha: float = 0.4, stack_x: bool = False, stack_y: bool = True) -> Learner:
#     "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."
#     if stack_y: learn.loss_func = MixUpLoss(learn.loss_func)
#     learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))
#     return learn

