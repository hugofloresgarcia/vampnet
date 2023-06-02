import argbind
from pathlib import Path
import yaml




"""example output: (yaml)

"""

@argbind.bind(without_prefix=True, positional=True)
def fine_tune(audio_file_or_folder: str, name: str):

    conf_dir = Path("conf")
    assert conf_dir.exists(), "conf directory not found. are you in the vampnet directory?"

    conf_dir = conf_dir / "generated"
    conf_dir.mkdir(exist_ok=True)

    finetune_dir = conf_dir / name
    finetune_dir.mkdir(exist_ok=True)

    finetune_c2f_conf = {
        "$include": ["conf/lora/lora.yml"],
        "fine_tune": True,
        "train/AudioLoader.sources": [audio_file_or_folder],
        "val/AudioLoader.sources": [audio_file_or_folder],
        "VampNet.n_codebooks": 14,
        "VampNet.n_conditioning_codebooks": 4,
        "VampNet.embedding_dim": 1280,
        "VampNet.n_layers": 16,
        "VampNet.n_heads": 20,
        "AudioDataset.duration": 3.0,
        "AudioDataset.loudness_cutoff": -40.0,
        "save_path": f"./runs/{name}/c2f",
    }

    finetune_coarse_conf = {
        "$include": ["conf/lora/lora.yml"],
        "fine_tune": True,
        "train/AudioLoader.sources": [audio_file_or_folder],
        "val/AudioLoader.sources": [audio_file_or_folder],
        "save_path": f"./runs/{name}/coarse",
    }

    interface_conf = {
        "Interface.coarse_ckpt": f"./runs/{name}/coarse/best/vampnet/weights.pth",
        "Interface.coarse2fine_ckpt": f"./runs/{name}/c2f/best/vampnet/weights.pth",
        "Interface.codec_ckpt": "./models/spotdl/codec.pth",
        "AudioLoader.sources": [audio_file_or_folder],
    }

    # save the confs
    with open(finetune_dir / "c2f.yml", "w") as f:
        yaml.dump(finetune_c2f_conf, f)

    with open(finetune_dir / "coarse.yml", "w") as f:
        yaml.dump(finetune_coarse_conf, f)
    
    with open(finetune_dir / "interface.yml", "w") as f: 
        yaml.dump(interface_conf, f)

    # copy the starter weights to the save paths
    import shutil

    def pmkdir(path):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        return path

    shutil.copy("./models/spotdl/c2f.pth", pmkdir(f"./runs/{name}/c2f/starter/vampnet/weights.pth"))
    shutil.copy("./models/spotdl/coarse.pth", pmkdir(f"./runs/{name}/coarse/starter/vampnet/weights.pth"))
    

    print(f"generated confs in {finetune_dir}. run training jobs with `python scripts/exp/train.py --args.load {finetune_dir}/<c2f/coarse>.yml --resume --load_weights --tag starter` ")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        fine_tune()



    