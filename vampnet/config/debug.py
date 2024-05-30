MODULE = 'vampnet'

CONFIG = 'debug-micrrrrro'
DB_FILE = "debug.db"

RESUME = False

# ~~ model params ~~
N_HEADS = 2
N_LAYERS = 2
N_CODEBOOKS = 4
LATENT_DIM = 8
EMBEDDING_DIM = 2048
DROPOUT = 0.0
CROSS_ATTEND_DIM = 1
MAX_SEQ_LEN = 512

# datasets
VAL_BATCH_SIZE = 256
BATCH_SIZE = 256
NUM_WORKERS = 16

VAL_FREQ = 500
SAMPLE_FREQ = 1000

# datasets
SEQ_LEN = 512
DATASET = "anns-animals"
CODES_KEY = "dac"
CTRL_KEYS = ["loudness"]

