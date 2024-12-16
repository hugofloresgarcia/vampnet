__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops", "cached_conv", "cached_conv.convs", "einops.layers.torch"]


from . import nn
from . import model
from . import utils
from .model import DAC
from .model import DACFile
