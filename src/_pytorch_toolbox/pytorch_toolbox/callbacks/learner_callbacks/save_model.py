import logging
from pathlib import Path

from pytorch_toolbox.defaults import Callable, Optional
from pytorch_toolbox.callbacks import TrackerCallback


class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(self, learn, save_path_creator: Optional[Callable], monitor: str = "VAL/total_loss",
                 mode: str = "auto", every: str = "improvement", file_name: str = "best_model", ):
        super().__init__(learn, monitor, mode)
        self.every = every
        if self.every not in ['improvement', 'epoch']:
            logging.warning(f'SaveModel every {every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        self.file_name = file_name
        self.save_path = Path(
            self.learn.path if save_path_creator is None else save_path_creator()) / f"{file_name}.pth"

    def on_epoch_end(self, epoch, phase, **kwargs) -> None:
        if self.every == "epoch":
            self.learn.save(f'{self.save_name}_{epoch}')
        else:  # every="improvement"
            current = self.get_monitor_value(epoch)
            if current is not None:
                self.best = current
                self.learn.save_model_with_path(f'{self.save_path}')
            elif self.operator(current, self.best):
                logging.info(f"The key to monitor: {self.monitor} has value: {current} but best value is {self.best}")
            else:
                logging.warning(f"The key to monitor: {self.monitor} is not found in the available keys: {list(self.learn.recorder.loss_history[(phase.name, epoch)].keys())}")


    def on_train_end(self, epoch, phase, **kwargs):
        last_epoch = epoch - 1
        current = self.get_monitor_value(last_epoch)
        if current is None:
            logging.warning(f"The key to monitor: {self.monitor} is not found in the available keys: {list(self.learn.recorder.loss_history[(phase.name, last_epoch)].keys())} so the best model will not be loaded")
            return
        if self.every == "improvement":
            try:
                self.learn.load_model_with_path(self.save_path)
            except FileNotFoundError:
                logging.info(f"File at {str(self.save_path)} is not found, skip loading model")
