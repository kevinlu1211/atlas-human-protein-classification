import numpy as np


def iafoss_training_scheme(learner, lr=2e-2):
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


def training_scheme_1(learner):
    lr = 2e-2
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=50, max_lr=lrs / 4)


def training_scheme_2(learner):
    lr = 2e-2
    lrs = [lr] * 3
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=50, max_lr=lrs)


def training_scheme_3(learner, lr=2e-3, epochs=50):
    lr = float(lr)
    lrs = np.array([lr / 10, lr / 3, lr])
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lrs / 4)
