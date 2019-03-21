from .csv_logger import CSVLogger
from .gradient_clipping import GradientClipping
from .lr_finder import LRFinder
from .mixed_precision import MixedPrecision
from .recorders import Recorder
from .reduce_lr_on_epoch_end import ReduceLROnEpochEndCallback
from .reduce_lr_on_plateau import ReduceLROnPlateauCallback
from .save_model import SaveModelCallback

learner_callback_lookup = {
    "CSVLogger": CSVLogger,
    "DeterminePhaseCallback": DeterminePhaseCallback,
    "GradientClipping": GradientClipping,
    "LRFinder": LRFinder,
    "MixedPrecision": MixedPrecision,
    "Recorder": Recorder,
    "ReduceLROnEpochEndCallback": ReduceLROnEpochEndCallback,
    "ReduceLROnPlateauCallback": ReduceLROnPlateauCallback,
    "SaveModelCallback": SaveModelCallback,
}