import numpy as np
import pytorch_toolbox.fastai.fastai as fastai

def three_tier_layer_group(learner):
    model = learner.model
    n_starting_layers = len(fastai.flatten_model(model[:6]))
    n_middle_layers = len(fastai.flatten_model(model[6:9]))
    # n_head = len(fastai.flatten_model(model[9:]))
    layer_groups = fastai.split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])
    return layer_groups

def iafoss_training_scheme(learner, lr=2e-2):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze_layer_groups(2)
    learner.fit(epochs=1, lr=[0, 0, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=2, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=4, max_lr=lrs / 4)
    learner.fit_one_cycle(cyc_len=8, max_lr=lrs / 16)


def training_scheme_1(learner, lr=2e-2, epochs=40):
    layer_groups = three_tier_layer_group(learner)
    learner.layer_groups = layer_groups
    lr = float(lr)
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs / 4)


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

def training_scheme_gapnet(learner, lr, epochs, div_factor=25.):
    lr = float(lr)
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)
