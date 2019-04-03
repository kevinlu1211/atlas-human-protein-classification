from .extract_label import LabelExtractorCallback
from .five_crop_tta import FiveCropTTACallback

callback_lookup = {
    "LabelExtractorCallback": LabelExtractorCallback,
    "FiveCropTTACallback": FiveCropTTACallback,
    "FiveCropTTAPredictionCallback": FiveCropTTACallback,
}