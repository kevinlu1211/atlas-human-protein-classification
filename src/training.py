import numpy as np
import pytorch_toolbox.fastai.fastai as fastai

def three_tier_layer_group(learner):
    model = learner.model
    n_starting_layers = len(fastai.flatten_model(model[:6]))
    n_middle_layers = len(fastai.flatten_model(model[6:8]))
    # n_head = len(fastai.flatten_model(model[9:]))
    layer_groups = fastai.split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
    return layer_groups

def iafoss_training_scheme(learner, lr=2e-2):
    learner.layer_groups = learner.model.layer_groups
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze_layer_groups(2)
    learner.fit(epochs=3, lr=[0, 0, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 8)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 8)
    learner.fit_one_cycle(cyc_len=8, max_lr=lrs / 16)

def warm_restarts_training_scheme(learner, lr=2e-2):
    lr = float(lr)
    learner.layer_groups = learner.model.layer_groups
    lrs = np.array([lr] * len(learner.layer_groups))
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs)
    learner.fit_one_cycle(cyc_len=6, max_lr=lrs / 2)
    learner.fit_one_cycle(cyc_len=8, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=10, max_lr=lrs / 8)
    learner.fit_one_cycle(cyc_len=12, max_lr=lrs / 16)
    learner.fit_one_cycle(cyc_len=14, max_lr=lrs / 20)



def training_scheme_1(learner, lr=2e-2, epochs=40):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs / 4)

def training_scheme_1_1(learner, lr=3e-4, epochs=40, pct_start=0.05):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = np.array([lr, lr, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs / 4, pct_start=pct_start)


def training_scheme_2(learner, lr=2e-2, epochs=50):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = [lr] * 3
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs)


def training_scheme_3(learner, lr=2e-3, epochs=50):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs / 4)

def training_scheme_4(learner, lr=2e-3, epochs=50, div_factor=25.):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs, div_factor=div_factor)

def training_scheme_one_cycle(learner, lr, epochs, div_factor=25):
    lr = float(lr)
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)

def training_scheme_gapnet_1(learner, lr, epochs, div_factor=25.):
    lr = float(lr)
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)

def training_scheme_lr_warmup(learner, epochs, warmup_epochs=None, lr=1e-3):
    lr = float(lr)
    start_lr = 1e-9
    div_factor = max(fastai.listify(lr)) / start_lr
    if warmup_epochs is None:
        warmup_epochs = int(epochs * 0.05) + 1
    learner.unfreeze()
    assert warmup_epochs < epochs
    pct_start = warmup_epochs / epochs
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, pct_start=pct_start, div_factor=div_factor)

def training_scheme_debug(learner, lr=1e-3, epochs=3, div_factor=25.):
    lr = float(lr)
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)

def training_scheme_se_resnext50_32x4d(learner, lr, epochs=50, div_factor=25):
    lr = float(lr)
    learner.layer_groups = learner.model.layer_groups
    learner.unfreeze()
    lrs = np.array([lr] * len(learner.model.layer_groups))
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs, div_factor=div_factor)



training_scheme_lookup = {
    "iafoss_training_scheme": iafoss_training_scheme,
    "training_scheme_one_cycle": training_scheme_one_cycle,
    "training_scheme_1": training_scheme_1,
    "training_scheme_1_1": training_scheme_1_1,
    "training_scheme_2": training_scheme_2,
    "training_scheme_3": training_scheme_3,
    "training_scheme_4": training_scheme_4,
    "training_scheme_gapnet_1": training_scheme_gapnet_1,
    "training_scheme_lr_warmup": training_scheme_lr_warmup,
    "training_scheme_debug": training_scheme_debug,
    "training_scheme_se_resnext50_32x4d": training_scheme_se_resnext50_32x4d,
    "warm_restarts_training_scheme": warm_restarts_training_scheme
}