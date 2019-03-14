from functools import partial
from numbers import Number
from typing import Optional

import numpy as np

from pytorch_toolbox.training import callbacks
from pytorch_toolbox.training.defaults import StartOptEnd, AnnealFunc
from pytorch_toolbox.utils import is_listy, is_tuple


