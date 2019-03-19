from functools import partial

from pytorch_toolbox.core.losses.soft_f_score_loss import calculate_soft_f_score_loss
from pytorch_toolbox.core.metrics.accuracy import accuracy
from pytorch_toolbox.core.metrics.focal_loss import focal_loss

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": partial(calculate_soft_f_score_loss, beta=1),
    "focal_loss": focal_loss
}