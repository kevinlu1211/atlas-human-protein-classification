from .extract_label import LabelExtractorCallback
from .five_crop_tta import FiveCropTTACallback
from .lr_one_cycle import OneCycleScheduler
from .lr_scheduler import GeneralScheduler, MultiStepScheduler

callback_lookup = {
    "GeneralScheduler": GeneralScheduler,
    "OneCycleScheduler": OneCycleScheduler,
    "MultiStepScheduler": MultiStepScheduler,
    "LabelExtractorCallback": LabelExtractorCallback,
    "FiveCropTTACallback": FiveCropTTACallback,
    "FiveCropTTAPredictionCallback": FiveCropTTACallback,
}