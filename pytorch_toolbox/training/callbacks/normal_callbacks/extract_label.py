from dataclasses import dataclass
from pytorch_toolbox.training.callbacks.core import Callback


@dataclass
class LabelExtractorCallback(Callback):
    label_key: str = 'label'

    def on_batch_begin(self, last_input, last_target, **kwargs):
        label = last_target.get(self.label_key)
        if label is not None:
            return last_input, last_target[self.label_key]
        else:
            return last_input, last_target
