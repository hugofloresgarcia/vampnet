MODULE = "vampnet"
CONFIG = "debug"

import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNS_DIR = "anns-animals-debug-4l"

# datasets
SEQ_LEN = 512
AUDIO_FOLDER = "/media/CHONK2/prosound_core_complete/Anns Animals"
DATASET = "anns-animals"
CODES_KEY = "dac"
CTRL_KEYS = ["rms"]
DB_FILE = "anns-animals.db"

# transforms
import audiotools as at
TFMS = {
    "rms": at.data.transforms.Compose([
        at.data.transforms.LowPass(cutoff=('uniform', 1, 10)),
    ])
}
# ~~ model params ~~
N_HEADS = 16
N_LAYERS = 4
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
D_CTRL = 1

DATASET = "anns-animals"
CONFIG = "anns-animals"

TRAIN_PROPORTION = 0.8
VAL_PROPORTION = 0.5
TEST_PROPORTION = 0.1

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
NUM_WORKERS = 0
VAL_IDX = [0, 1, 2, 3, 4]
SAVE_ITERS = [50_000, 100_000, 200_000, 400_000, 800_000, 1_000_000]
NUM_ITERS = 250_000

