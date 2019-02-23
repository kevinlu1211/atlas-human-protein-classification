from pytorch_toolbox.utils import listify
# import pytorch_toolbox.fastai.fastai as fastai


def training_scheme_one_cycle(learner, lr, epochs, div_factor=25):
    lr = [float(lr_) for lr_ in listify(lr)]
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)


def training_scheme_fit(learner, lr, epochs):
    lr = [float(lr_) for lr_ in listify(lr)]
    if learner.model.layer_groups is not None:
        learner.layer_groups = learner.model.layer_groups
    learner.unfreeze()
    learner.layer_groups = learner.model.layer_groups
    learner.fit(epochs=epochs, lr=lr)


def training_scheme_lr_warmup(learner, epochs, warmup_epochs=None, lr=1e-3):
    lr = float(lr)
    start_lr = 1e-9
    div_factor = max(listify(lr)) / start_lr
    if warmup_epochs is None:
        warmup_epochs = int(epochs * 0.05) + 1
    learner.unfreeze()
    assert warmup_epochs < epochs
    pct_start = warmup_epochs / epochs
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, pct_start=pct_start, div_factor=div_factor)


training_scheme_lookup = {
    "training_scheme_one_cycle": training_scheme_one_cycle,
    "training_scheme_fit": training_scheme_fit,
    "training_scheme_lr_warmup": training_scheme_lr_warmup,
}
