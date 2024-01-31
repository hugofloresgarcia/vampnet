from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict
import subprocess

import argbind
""""
this script is trash!
"""
@argbind.bind(without_prefix=True)
def main(
    audio_file_or_folder: str = None,
):
    assert audio_file_or_folder is not None
    audio_file_or_folder = Path(audio_file_or_folder)
    name = audio_file_or_folder.stem

    csv_path = Path("data/metadata") / f"{name}.csv"

    # create a dataset
    print(f"Creating dataset from  {audio_file_or_folder}")
    print(f"Saving to {csv_path}")
    subprocess.check_output([
        "python", "scripts/pre/create_dataset.py", 
        f"--audio_folder", f"{audio_file_or_folder}",
        f"--output_file", f"{csv_path}"
    ])

    # output folder should be data/fine-tuned/{name}
    codes_folder = Path("data/codes-unchunked") / name
    if "prosound_redacted" in audio_file_or_folder:
        # load the metadat csv with pandas
        metadata = pd.read_csv(csv_path)
        # change the audio root to prosound_redacted

        print(f"WARNING: using prosound_redacted, changing codes_folder")
        codes_folder = codes_folder.parent / "prosound_redacted"

    # encode with dac
    print(f"Encoding dataset with dac")
    subprocess.check_output([
        "python", "scripts/pre/condition_workers.py", 
        f"--input_csv={csv_path}",
        f"--output_folder={codes_folder}",
        "--conditioner_name=dac", 
    ])

    # make a fine tune conf
    print(f"Making fine tune conf")
    ftdir = Path("conf/fine-tune")
    lora_conf = Path("conf/lora.yaml")
    import yaml
    with open(lora_conf, "r") as f:
        conf = yaml.safe_load(f)
    
    conf["train/DACDataset.split"] = None
    conf["val/DACDataset.split"] = None
    conf["val/DACDataset.length"] = 100

    conf["DACDataset.metadata_csvs"] = [str(csv_path)]

    conf["save_path"] = f"runs/fine-tuned/{name}/"

    # save the conf
    ftdir.mkdir(parents=True, exist_ok=True)
    conf_path = ftdir / f"{name}.yaml"
    with open(conf_path, "w") as f:
        yaml.dump(conf, f)
    
    print(f"done! run the following command to fine tune:")
    print(f"python scripts/exp/train.py --args.load={conf_path} --amp --compile")


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()
    


