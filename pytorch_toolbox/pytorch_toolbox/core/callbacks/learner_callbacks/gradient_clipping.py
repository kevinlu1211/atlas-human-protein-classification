import torch.nn as nn

from pytorch_toolbox.core import LearnerCallback


class GradientClipping(LearnerCallback):
    "To do gradient clipping during training."
    clip: float

    def __init__(self, learn, clip=1.0):
        super().__init__(learn)
        self.clip = clip

    def on_backward_end(self, **kwargs):
        if self.clip:
            nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)
