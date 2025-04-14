
from . import modules
from pathlib import Path
from . import scheduler
from .interface import Interface
from .modules.transformer import VampNet


__version__ = "0.0.1"

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "vampnet"

from huggingface_hub import hf_hub_download, HfFileSystem
DEFAULT_HF_MODEL_REPO_DIR = ROOT / "DEFAULT_HF_MODEL_REPO"
DEFAULT_HF_MODEL_REPO = DEFAULT_HF_MODEL_REPO_DIR.read_text().strip()
# DEFAULT_HF_MODEL_REPO = "hugggof/vampnet"
FS = HfFileSystem()

def download_codec():
    # from dac.model.dac import DAC
    from lac.model.lac import LAC as DAC
    repo_id = DEFAULT_HF_MODEL_REPO
    filename = "codec.pth"
    codec_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=None, 
        local_dir=MODELS_DIR
    )
    return codec_path
    

def download_default():
    filenames = ["coarse.pth", "c2f.pth", "wavebeat.pth"]
    repo_id = DEFAULT_HF_MODEL_REPO
    paths = []
    for filename in filenames:
        path = f"{MODELS_DIR}/{filename}"
        if not Path(path).exists():
            print(f"{path} does not exist, downloading")
            FS.download(f"{repo_id}/{filename}", path)
        paths.append(path)
    
    # load the models
    return paths[0], paths[1]


def download_finetuned(name, repo_id=DEFAULT_HF_MODEL_REPO):
    filenames = ["coarse.pth", "c2f.pth"]
    paths = []
    for filename in filenames:
        path = f"{MODELS_DIR}/loras/{name}/{filename}"
        if not Path(path).exists():
            print(f"{path} does not exist, downloading")
            FS.download(f"{repo_id}/loras/{name}/{filename}", path)
        paths.append(path)
    
    # load the models
    return paths[0], paths[1]
    
def list_finetuned(repo_id=DEFAULT_HF_MODEL_REPO):
    diritems = FS.listdir(f"{repo_id}/loras")
    # iterate through all the names
    valid_diritems = []
    for item in diritems:
        model_file_items = FS.listdir(item["name"])
        item_names = [item["name"].split("/")[-1] for item in model_file_items]
        # check that theres a "c2f.pth" and "coarse.pth" in the items
        c2f_exists = "c2f.pth" in item_names
        coarse_exists = "coarse.pth" in item_names
        if c2f_exists and coarse_exists:
            valid_diritems.append(item)

    # get the names of the valid items
    names = [item["name"].split("/")[-1] for item in valid_diritems]
    return names


