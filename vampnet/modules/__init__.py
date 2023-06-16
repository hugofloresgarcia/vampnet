import audiotools

audiotools.ml.BaseModel.INTERN += ["vampnet.modules.**"]
audiotools.ml.BaseModel.EXTERN += ["einops", "flash_attn.flash_attention", "loralib", "x_transformers"]

from .transformer import VampNet
