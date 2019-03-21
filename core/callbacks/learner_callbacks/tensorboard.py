from pathlib import Path

from tensorboardX import SummaryWriter

from pytorch_toolbox.core.defaults import Callable, Optional
from pytorch_toolbox.core.callbacks import LearnerCallback


class TensorBoardRecorder(LearnerCallback):
    def __init__(self, learn, save_path_creator: Optional[Callable], file_name="tensorboard_logs"):
        super().__init__(learn)
        self.tb_writer = SummaryWriter
        self.log_path = Path(
            self.learn.path if save_path_creator is None else save_path_creator()) / f"{file_name}.log"






