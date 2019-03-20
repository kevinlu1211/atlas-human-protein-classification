from pytorch_toolbox.core.utils import Phase
from pytorch_toolbox.core.callbacks import LearnerCallback


class DeterminePhaseCallback(LearnerCallback):
    _order = -15

    def __init__(self, learn, label_key: str = 'label'):
        super().__init__(learn)
        self.label_key = label_key

    def on_batch_begin(self, train, last_target, **kwargs):
        self.learn.phase = self.determine_phase(train, last_target)

    def determine_phase(self, train, last_target):
        if train:
            return Phase.TRAIN
        else:
            label = last_target.get(self.label_key)
            if label is not None:
                return Phase.VAL
            else:
                return Phase.TEST
