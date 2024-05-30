MODULE = "vampnet"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"