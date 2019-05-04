from .csv_logger import CSVLogger
from .gradient_clipping import GradientClipping
from .lr_finder import LRFinder

from .mixed_precision import MixedPrecision
from .recorders import Recorder, TensorBoardRecorder
from .reduce_lr_on_epoch_end import ReduceLROnEpochEndCallback
from .reduce_lr_on_plateau import ReduceLROnPlateauCallback
from .save_model import SaveModelCallback
from .hooks import HookCallback

learner_callback_lookup = {
    "CSVLogger": CSVLogger,
    "GradientClipping": GradientClipping,
    "LRFinder": LRFinder,
    "MixedPrecision": MixedPrecision,
    "Recorder": Recorder,
    "TensorBoardRecorder": TensorBoardRecorder,
    "ReduceLROnEpochEndCallback": ReduceLROnEpochEndCallback,
    "ReduceLROnPlateauCallback": ReduceLROnPlateauCallback,
    "SaveModelCallback": SaveModelCallback,
}