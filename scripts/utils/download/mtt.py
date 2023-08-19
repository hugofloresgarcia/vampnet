
import argbind
from pathlib import Path
from typing import Union
import subprocess
import pandas as pd

from scripts.utils.download import download_file, download, unzip


mtt_files = [
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001",
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002",
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003",
]

metadata_url = "https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv"

@argbind.bind(without_prefix=True)
def download_mtt(
    dest: str = "~/data/mtt",
):
    dest_dir = Path(dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # download the dataset
    zipnames = [download(url, dest_dir) for url in mtt_files]

    # unzip
    unzip(zipnames[0], dest_dir)

    # download the metadata
    print(f"downloading {metadata_url} to {dest_dir}")
    meta_path = download(metadata_url, dest_dir)

    # read the metadata
    df = pd.read_csv(meta_path)
    print(f"metadata has {len(df)} rows")
    
    print(f"we're done here!")
    print(f"you can create a train test split by running the following command:")
    print(f"python scripts/utils/split.py --folder {dest_dir} --test_size 0.1 --pattern '**/*.mp3'")

    
if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        download_mtt()