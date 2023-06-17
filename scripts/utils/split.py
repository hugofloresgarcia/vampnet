from pathlib import Path
import random
import shutil
import os
import json 

import argbind
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from audiotools.core import util


@argbind.bind(without_prefix=True)
def train_test_split(
    audio_folder: str = ".", 
    test_size: float = 0.2,
    seed: int = 42,
    pattern: str = "**/*.mp3",
):
    print(f"finding audio")

    audio_folder = Path(audio_folder)
    audio_files = list(tqdm(audio_folder.glob(pattern)))
    print(f"found {len(audio_files)} audio files")
    
    # split according to test_size
    n_test = int(len(audio_files) * test_size)
    n_train = len(audio_files) - n_test

    # shuffle
    random.seed(seed)
    random.shuffle(audio_files)

    train_files = audio_files[:n_train]
    test_files = audio_files[n_train:]


    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    continue_ = input("Continue [yn]? ") or "n"

    if continue_ != "y":
        return
    
    for split, files in (
        ("train", train_files), ("test", test_files)
    ):
        for file in tqdm(files):
            out_file = audio_folder.parent / f"{audio_folder.name}-{split}" / Path(file).name
            out_file.parent.mkdir(exist_ok=True, parents=True)
            os.symlink(file, out_file)

        # save split as json
        with open(Path(audio_folder) / f"{split}.json", "w") as f:
            json.dump([str(f) for f in files], f)
    

    
if __name__ == "__main__":
    args  = argbind.parse_args()

    with argbind.scope(args):
        train_test_split()