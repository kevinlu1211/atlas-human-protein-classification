from .lr_finder import LRFinder
from .csv_logger import CSVLogger
from .reduce_lr_on_epoch_end import ReduceLROnEpochEndCallback
from .reduce_lr_on_plateau import ReduceLROnPlateauCallback
from .save_model import SaveModelCallback

learner_callback_lookup = {
    "CSVLogger": CSVLogger,
    "LRFinder": LRFinder,
    "ReduceLROnEpochEndCallback": ReduceLROnEpochEndCallback,
    "ReduceLROnPlateauCallback": ReduceLROnPlateauCallback,
    "SaveModelCallback": SaveModelCallback
}