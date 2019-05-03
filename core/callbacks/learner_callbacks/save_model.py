import logging
from pathlib import Path

from core.defaults import Callable, Optional
from core.callbacks import TrackerCallback


class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(self, learn, save_path_creator: Optional[Callable], monitor: str = "val_loss",
                 mode: str = "auto", every: str = "improvement", file_name: str = "best_model", ):
        super().__init__(learn, monitor, mode)
        self.every = every
        if self.every not in ['improvement', 'epoch']:
            logging.warning(f'SaveModel every {every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        self.file_name = file_name
        self.save_path = Path(
            self.learn.path if save_path_creator is None else save_path_creator()) / f"{file_name}.pth"

    def on_epoch_end(self, epoch, **kwargs) -> None:
        if self.every == "epoch":
            self.learn.save(f'{self.save_name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value(epoch)
            if current is not None and self.operator(current, self.best):
                self.best = current
                self.learn.save_model_with_path(f'{self.save_path}')

    def on_train_end(self, epoch, **kwargs):
        current = self.get_monitor_value(epoch)
        if current is None:
            return
        if self.every == "improvement":
            try:
                self.learn.load_model_with_path(self.save_path)
            except FileNotFoundError:
                logging.info(f"File at {str(self.save_path)} is not found, skip loading model")
