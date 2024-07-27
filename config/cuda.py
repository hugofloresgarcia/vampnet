MODULE = "vampnet"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ~~ model params ~~
N_HEADS = 16
N_LAYERS = 16
N_CODEBOOKS = 4
N_CONDITIONING_CODEBOOKS = 0
LATENT_DIM = 8
EMBEDDING_DIM = 1280
VOCAB_SIZE = 1024
DROPOUT = 0.0
CROSS_ATTEND_DIM = 0
MAX_SEQ_LEN = 1024
NUM_REG_TOKENS = 0
LORA_R = 8

