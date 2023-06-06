import argbind
from pathlib import Path
import yaml
from typing import List




"""example output: (yaml)

"""

@argbind.bind(without_prefix=True, positional=True)
def fine_tune(audio_files_or_folders: List[str], name: str):

    conf_dir = Path("conf")
    assert conf_dir.exists(), "conf directory not found. are you in the vampnet directory?"

    conf_dir = conf_dir / "generated"
    conf_dir.mkdir(exist_ok=True)

    finetune_dir = conf_dir / name
    finetune_dir.mkdir(exist_ok=True)

    finetune_c2f_conf = {
        "$include": ["conf/lora/lora.yml"],
        "fine_tune": True,
        "train/AudioLoader.sources": audio_files_or_folders,
        "val/AudioLoader.sources": audio_files_or_folders,
        "VampNet.n_codebooks": 14,
        "VampNet.n_conditioning_codebooks": 4,
        "VampNet.embedding_dim": 1280,
        "VampNet.n_layers": 16,
        "VampNet.n_heads": 20,
        "AudioDataset.duration": 3.0,
        "AudioDataset.loudness_cutoff": -40.0,
        "save_path": f"./runs/{name}/c2f",
        "fine_tune_checkpoint": "./models/spotdl/c2f.pth"
    }

    finetune_coarse_conf = {
        "$include": ["conf/lora/lora.yml"],
        "fine_tune": True,
        "train/AudioLoader.sources": audio_files_or_folders,
        "val/AudioLoader.sources": audio_files_or_folders,
        "save_path": f"./runs/{name}/coarse",
        "fine_tune_checkpoint": "./models/spotdl/coarse.pth"
    }

    interface_conf = {
        "Interface.coarse_ckpt": f"./models/spotdl/coarse.pth",
        "Interface.coarse_lora_ckpt": f"./runs/{name}/coarse/latest/lora.pth",

        "Interface.coarse2fine_ckpt": f"./models/spotdl/c2f.pth",
        "Interface.coarse2fine_lora_ckpt": f"./runs/{name}/c2f/latest/lora.pth",

        "Interface.codec_ckpt": "./models/spotdl/codec.pth",
        "AudioLoader.sources": [audio_files_or_folders],
    }

    # save the confs
    with open(finetune_dir / "c2f.yml", "w") as f:
        yaml.dump(finetune_c2f_conf, f)

    with open(finetune_dir / "coarse.yml", "w") as f:
        yaml.dump(finetune_coarse_conf, f)
    
    with open(finetune_dir / "interface.yml", "w") as f: 
        yaml.dump(interface_conf, f)


    print(f"generated confs in {finetune_dir}. run training jobs with `python scripts/exp/train.py --args.load {finetune_dir}/<c2f/coarse>.yml` ")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        fine_tune()



    