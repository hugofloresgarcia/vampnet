MODULE = "vampnet"

CONFIG = "example"
DB_FILE = "example.db"
DATASET = "example"

NUM_WORKERS = 0
BATCH_SIZE = 2
import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ~~ model ~~
CROSS_ATTEND_DIM = 0 


# ~ huggingface export ~
HF_USERNAME = "hugggof"
HF_REPO_NAME = "vampnet-example"
EXPORT_MODEL_TAG = "best"
MODEL_EXT = ".vampnet"