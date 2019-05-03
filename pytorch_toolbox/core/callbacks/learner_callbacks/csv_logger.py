from pathlib import Path

import pandas as pd

from core.defaults import StrList, Any, Tensor, MetricsList, Optional, Callable
from core.callbacks import LearnerCallback
from core.utils import if_none


class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."

    def __init__(self, learn, save_path_creator: Optional[Callable] = None, file_name: str = "history"):
        super().__init__(learn)
        self.learn = learn
        path = Path(self.learn.path if save_path_creator is None else save_path_creator())
        self.file_name = file_name
        self.save_path = path / f'{self.file_name}.csv'

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.save_path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.save_path.open('w')
        self.file.write(','.join(self.learn.recorder.names) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        last_metrics = if_none(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:
        self.file.close()
