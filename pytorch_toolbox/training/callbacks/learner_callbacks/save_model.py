import logging
from dataclasses import dataclass

from pytorch_toolbox.training.callbacks.core import TrackerCallback
from pytorch_toolbox.training.defaults import Callable


@dataclass
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every: str = 'improvement'
    name: str = 'best_model'

    # need to create a default value to get around the error TypeError: non-default argument 'save_path_creator' follows default argument
    save_path_creator: Callable = None

    def __post_init__(self):
        assert self.save_path_creator is not None
        if self.every not in ['improvement', 'epoch']:
            logging.warning(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        self.save_path = self.save_path_creator()
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs) -> None:
        if self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value(epoch)
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save(f'{self.save_path / self.name}')

    def on_train_end(self, epoch, **kwargs):
        current = self.get_monitor_value(epoch)
        if current is None:
            return
        if self.every == "improvement":
            self.learn.load(f'{self.save_path / self.name}')
