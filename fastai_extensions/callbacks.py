import sys
sys.path.append("../fastai")

import torch

import pytorch_toolbox.fastai.fastai as fastai
import dataclasses

@dataclasses.dataclass
class NameExtractionTrainer(fastai.Callback):
    label_key: str = 'label'

    def on_batch_begin(self, last_input, last_target, **kwargs):
        label = last_target.get(self.label_key)
        if label is not None:
            return last_input, last_target[self.label_key]
        else:
            return last_input, last_target
