from dataclasses import dataclass

import pandas as pd

from pytorch_toolbox.training.callbacks.core import TrackerCallback
from pytorch_toolbox.training.defaults import Callable

@dataclass
class ReduceLROnEpochEndCallback(TrackerCallback):
    wait_duration: int = 10
    save_path_creator: Callable = None

    def __post_init__(self):
        assert self.save_path_creator is not None
        self.lr_history = []
        self.save_path = self.save_path_creator()
        super().__post_init__()

    def _wait_for_user_prompt_to_change_lr(self):
        print(f"Waiting for {self.wait_duration} seconds keyboard interrupt to change LR")
        time.sleep(self.wait_duration)

    def _user_input_prompt_for_new_lr(self):
        new_lr = input(
            "Please type in the new lr, if it is a list of lrs separate them by spaces,"
            " type n or no to continue training without changing lr\n")
        if new_lr.lower() in ["n", "no"]:
            return None
        new_lr = new_lr.split()
        new_lr = [float(lr) for lr in new_lr]
        return new_lr

    def _request_user_input_for_new_lr(self):
        current_lr = self.learn.opt.read_val('lr')
        print(f"The current LR is: {current_lr}")
        while True:
            try:
                new_lr = self._user_input_prompt_for_new_lr()
                if new_lr is None:
                    return
            except Exception as e:
                print(e)

    def on_epoch_end(self, epoch, **kwargs):
        current_lr = self.learn.opt.read_val('lr')
        if epoch == 0:
            self.lr_history.append(dict(epoch=epoch, lr=current_lr))
        try:
            self._wait_for_user_prompt_to_change_lr()
        except KeyboardInterrupt:
            self._request_user_input_for_new_lr()

    def on_train_end(self, **kwargs):
        lr_history_df = pd.DataFrame(self.lr_history)
        lr_history_df.to_csv(self.save_path / "lr_history.csv")