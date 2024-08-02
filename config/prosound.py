MODULE = 'vampnet'
CONFIG = "prosound"

RUNS_DIR = "runs/norms-d1280-prosound-8h-rot"

# ~~ debug ~~
VERBOSE = False

# ~~ data ~~ 
from pathlib import Path
AUDIO_FOLDER = "/media/CHONK2/prosound_core_complete"

# ~~ tensors ~~
import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# datasets
SEQ_LEN = 512
DATASET = "prosound"
CODES_KEY = "dac"
CTRL_KEYS = []
DB_FILE = "prosound.db"
AUDIO_LOOKUP_MAX_AUDIO_CHANNELS = 2

# ~~ splits ~~~
TRAIN_PROPORTION = 0.8
VAL_PROPORTION = 0.1
TEST_PROPORTION = 0.1

# ~~ model params ~~
N_HEADS = 8
N_LAYERS = 8
N_CODEBOOKS = 9
N_CONDITIONING_CODEBOOKS = 0
LATENT_DIM = 8
EMBEDDING_DIM = 1280
VOCAB_SIZE = 1024
DROPOUT = 0.0
CROSS_ATTEND_DIM = 0
MAX_SEQ_LEN = 1024
NUM_REG_TOKENS = 0
LORA_R = 0
D_CTRL = 0


# ~~~ training ~~~
IGNORE_INDEX = -100

# ~ huggingface export ~
HF_USERNAME = "hugggof"
HF_REPO_NAME = "vampnet-models"
EXPORT_MODEL_TAG = "best"
MODEL_EXT = ".vampnet"

# TODO: this flag should be tied to a checkpoint, 
# yet it is not. how can we amend this?
SCHEDULE = "cosine"

## captioning
NUM_CAPTIONS = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SEED = 102345
AMP = True
COMPILE = False

# optimizers
LR = 0.001

SCHED_FACTOR = 2.0
SCHED_WARMUP = 10_000

GRAD_ACC_STEPS = 1
GRAD_CLIP_VAL = 1.0

# datasets
VAL_BATCH_SIZE = 36
BATCH_SIZE = 36

VAL_FREQ = 1000
SAMPLE_FREQ = 3000

import os
# NUM_WORKERS = 16
VAL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
SAVE_ITERS = [50_000, 100_000, 200_000, 400_000, 800_000, 1_000_000]
NUM_ITERS = 250_000

