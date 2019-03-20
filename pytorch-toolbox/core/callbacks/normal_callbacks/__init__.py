from .extract_label import LabelExtractorCallback
from .five_crop_tta import FiveCropTTACallback
from .one_cycle import OneCycleScheduler

callback_lookup = {
    "LabelExtractorCallback": LabelExtractorCallback,
    "FiveCropTTACallback": FiveCropTTACallback,
    "FiveCropTTAPredictionCallback": FiveCropTTACallback,
    "OneCycleScheduler": OneCycleScheduler
}