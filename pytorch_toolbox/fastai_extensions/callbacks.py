import sys
sys.path.append("../fastai")

import torch

from fastai import *
import fastai

@dataclass
class NameExtractionTrainer(fastai.Callback):
    label_key: str = 'label'

    def on_batch_begin(self, last_input, last_target, **kwargs):
        label = last_target.get(self.label_key)
        if label is not None:
            return last_input, last_target[self.label_key]
        else:
            return last_input, last_target

@dataclass
class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    clip: float = 1.0

    def on_backward_end(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)