from .csv_logger import CSVLogger
from .determine_phase import DeterminePhaseCallback
from .gradient_clipping import GradientClipping
from .lr_finder import LRFinder
from pytorch_toolbox.core.callbacks.learner_callbacks.lr_one_cycle import OneCycleScheduler
from pytorch_toolbox.core.callbacks.learner_callbacks.lr_scheduler import GeneralScheduler, MultiStepScheduler
from .mixed_precision import MixedPrecision
from .recorders import Recorder, TensorBoardRecorder
from .reduce_lr_on_epoch_end import ReduceLROnEpochEndCallback
from .reduce_lr_on_plateau import ReduceLROnPlateauCallback
from .save_model import SaveModelCallback

learner_callback_lookup = {
    "CSVLogger": CSVLogger,
    "DeterminePhaseCallback": DeterminePhaseCallback,
    "GeneralScheduler": GeneralScheduler,
    "GradientClipping": GradientClipping,
    "LRFinder": LRFinder,
    "OneCycleScheduler": OneCycleScheduler,
    "MixedPrecision": MixedPrecision,
    "MultiStepScheduler": MultiStepScheduler,
    "Recorder": Recorder,
    "TensorBoardRecorder": TensorBoardRecorder,
    "ReduceLROnEpochEndCallback": ReduceLROnEpochEndCallback,
    "ReduceLROnPlateauCallback": ReduceLROnPlateauCallback,
    "SaveModelCallback": SaveModelCallback,
}