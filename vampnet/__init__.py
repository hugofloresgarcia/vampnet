###############################################################################
# Configuration
###############################################################################

# read our version
from pathlib import Path

# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('vampnet', defaults)

# Import configuration parameters
del defaults
from .config.defaults import *
from .config.static import *


###############################################################################
# Module imports
###############################################################################


# Your imports go here
from . import controls
from .controls.codec import load_codec
from .model import transformer
from . import interface
from . import db
from . import util
from . import export
from . import train
from . import fine_tune
from . import mask
from . import signal

from pathlib import Path
from huggingface_hub import hf_hub_download

# TODO: load dac should download dac
# TODO: fix COLAB NOTEBOOK NT WORKING
# TODO: models dir should be there. 
# TODO: rename to vampnet. 

HF_MODELS = [
    "hugggof/vampnet-models:spotdl-8l-d1280-best",
    "hugggof/vampnet-models:vampnet-base-best", 
]
DEFAULT_MODEL = HF_MODELS[0]
# download a model from the huggingface hub
def load_hub_model(model_id = HF_MODELS[0]):
    repo_id, model_name = model_id.split(":")
    # download the model
    filename = Path(model_name).with_suffix(vampnet.MODEL_EXT)
    # filename = vampnet.MODEL_FILE.name
    subfolder = vampnet.MODEL_FILE.parent.relative_to(vampnet.ROOT)
    model_dir = hf_hub_download(
        repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=MODELS_DIR / "hub_cache")
    # load the model
    return transformer.VampNet.load(str(model_dir))
    
    
def list_hub_models():
    # list all models on the huggingface hub
    return HF_MODELS
    

def list_local_models():
    # list all .vampnet files in the models directory
    return [p.stem for p in MODELS_DIR.glob(f'*{vampnet.MODEL_EXT}')]
    pass


def load_local_model(name):
    # load a model by name
    ckpt = MODELS_DIR / f"{name}.vampnet"
    return transformer.VampNet.load(ckpt)


def save_model(model, name):
    # save a model to the models directory
    ckpt = MODELS_DIR / f"{name}.vampnet"
    model.save(ckpt)
    return ckpt

def load_model(name):
    if name in list_local_models():
        print(f"loading model {name} from local models.")
        return load_local_model(name)
    elif ("/" in name) and (":" in name):
        print(f"loading model {name} from the huggingface hub.")
        # breakpoint()
        return load_hub_model(name)
    else:
        raise ValueError(f"Model {name} not found in local models or on the hub. available: \nlocal\n{list_local_models()} \nhub\n{list_hub_models()}")

