from functools import partial

from .soft_f_score import soft_f_score
from .accuracy import accuracy
from .focal_loss import focal_loss

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": partial(soft_f_score, beta=1),
    "focal_loss": focal_loss
}