
import logging
import math
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import audiotools as at
import loralib as lora
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers import ContinuousTransformerWrapper
from x_transformers import Encoder

from ..mask import _gamma, scalar_to_batch_tensor, full_mask
from ..util import codebook_flatten
from ..util import codebook_unflatten
from .layers import CodebookEmbedding
from .layers import WNConv1d
