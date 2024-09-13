import audiotools

audiotools.ml.BaseModel.INTERN += ["vampnet.modules.**"]
audiotools.ml.BaseModel.EXTERN += ["einops", "flash_attn.flash_attention", "loralib"]

from .transformer import VampNet