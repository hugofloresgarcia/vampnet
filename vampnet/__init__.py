
from . import modules
from pathlib import Path
from . import scheduler
from .modules.transformer import VampNet
from . import interface

__version__ = "0.0.1"

# TODO: all of the code below should be updated

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "vampnet" / __version__

SCRATCH_DIR = Path("scratch/")
SCRATCH_DIR.mkdir(exist_ok=True, parents=True)