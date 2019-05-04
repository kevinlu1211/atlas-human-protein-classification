from dataclasses import dataclass

import numpy as np

from pytorch_toolbox.callbacks import TrackerCallback


@dataclass
class ReduceLROnPlateauCallback(TrackerCallback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    def __init__(self, learn, patient: int = 0, factor: float = 0.2, min_delta: int = 0):
        super.__init__(learn)
        self.patient = patient
        self.factor = factor
        self.min_delta = min_delta

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:  self.min_delta *= -1

    def on_train_begin(self, **kwargs) -> None:
        self.wait, self.opt = 0, self.learn.opt
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs) -> None:
        current = self.get_monitor_value(epoch)
        if current is None: return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.opt.lr *= self.factor
                self.wait = 0
                print(f'Epoch {epoch}: reducing lr to {self.opt.lr}')
