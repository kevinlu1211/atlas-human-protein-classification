from functools import partial

from .soft_f_score import soft_f_score_metric
from .accuracy import accuracy
from .focal_loss import focal_loss_metric

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": partial(soft_f_score_metric, beta=1),
    "focal_loss": focal_loss_metric
}