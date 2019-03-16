from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from pytorch_toolbox.training.defaults import StrList, Any, Tensor, MetricsList, Optional
from pytorch_toolbox.training.callbacks.core import LearnerCallback
from pytorch_toolbox.utils import if_none


@dataclass
class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."
    file_save_path: Optional[str] = None
    file_name: str = 'history'

    def __post_init__(self):
        super().__post_init__()
        path = Path(if_none(self.file_save_path, self.learn.path))
        self.file_path_name = path / f'{self.file_name}.csv'

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.learn.recorder.names) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        last_metrics = if_none(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:
        self.file.close()
