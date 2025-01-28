import audiotools

audiotools.ml.BaseModel.INTERN += ["vampnet.lac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model

