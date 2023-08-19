"""
usage: 

```
python scripts/utils/download/fma.py --dest ~/data/fma --size small
```
"""
import argbind
from pathlib import Path
from typing import Union
import subprocess

from scripts.utils.download import download_file, download, unzip


URLS = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
    "full": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",
    "metadata": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}

@argbind.bind(without_prefix=True)
def download_fma(
    dest: str = "~/data/fma",
    size: str = "small",
):
    dest_dir = Path(dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # download the dataset
    assert size in URLS.keys(), f"size must be one of {URLS.keys()}"
    url = URLS[size]

    data_zip = download(url, dest_dir)
    unzip(data_zip, dest_dir, )

    meta_zip = download(URLS["metadata"], dest_dir)
    unzip(meta_zip, dest_dir)

    print(f"we're done here!")
    print(f"you can create a train test split by running the following command:")
    print(f"python scripts/utils/split.py --folder {dest_dir/f'fma_{size}'} --test_size 0.1 --pattern '**/*.mp3'")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        download_fma()

    