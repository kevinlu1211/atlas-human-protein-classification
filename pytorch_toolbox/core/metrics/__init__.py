from functools import partial

from pytorch_toolbox.core.metrics.soft_f_score import soft_f_score
from pytorch_toolbox.core.metrics.accuracy import accuracy
from pytorch_toolbox.core.metrics.focal_loss import focal_loss

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": partial(soft_f_score, beta=1),
    "focal_loss": focal_loss
}