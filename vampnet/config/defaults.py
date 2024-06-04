MODULE = 'vampnet'

CONFIG = "default"
VERSION = "0.0.1"

# ~~ debug ~~
VERBOSE = False
RESUME = False

# ~~ data ~~ 
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
DB_FILE = "vamp.db"

MODELS_DIR = ROOT / "models"

# ~~ audio ~~
HOP_SIZE = 512
SAMPLE_RATE = 44100
CODEC_PATH = MODELS_DIR / "codec.pth"
LOUD_NORM = -16 # all audio is normalized to this by the codec

# ~~ tensors ~~
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SEED = 10110100
AMP = True
COMPILE = True

# optimizers
LR = 0.001

SCHED_FACTOR = 2.0
SCHED_WARMUP = 10_000

GRAD_ACC_STEPS = 1
GRAD_CLIP_VAL = 10.0

# datasets
VAL_BATCH_SIZE = 36
BATCH_SIZE = 36

VAL_FREQ = 1000
SAMPLE_FREQ = 10_000

import os
# NUM_WORKERS = 16
NUM_WORKERS = 3 * (os.cpu_count() // 4)
VAL_IDX = [0, 1, 2, 3, 4]
SAVE_ITERS = [50_000, 100_000, 200_000, 400_000, 800_000, 1_000_000]
NUM_ITERS = 250_000

# datasets
SEQ_LEN = 512
DATASET = "anns-animals"
CODES_KEY = "dac"
CTRL_KEYS = []

TRAIN_PROPORTION = 0.8
VAL_PROPORTION = 0.1
TEST_PROPORTION = 0.1

IGNORE_INDEX = -100

# ~ huggingface export ~
HF_USERNAME = "hugggof"
HF_REPO_NAME = "vampnet-models"
EXPORT_MODEL_TAG = "best"
MODEL_EXT = ".vampnet"

# TODO: this flag should be tied to a checkpoint, 
# yet it is not. how can we amend this?
SCHEDULE = "linear"