from pathlib import Path
import random
import shutil
import os
import json

import argbind
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from audiotools.core import util

# TODO: make this so that it operates on a metadata csv instead of a folder
@argbind.bind(without_prefix=True)
def train_test_val_split(
    folder: str = ".",
    test_size: float = 0.2,
    val_size: float = 0.1,  # New validation size argument
    pattern: str = "**/*.wav",
    seed: int = 42,
    out_dir: str = ".",
):
    print(f"finding audio")
    audio_folder: Path = Path(folder)
    audio_files = list(audio_folder.glob(pattern))
    print(f"found {len(audio_files)} audio files")

    # split according to test_size and val_size
    n_test = int(len(audio_files) * test_size)
    n_val = int(len(audio_files) * val_size) # New line
    n_train = len(audio_files) - n_test - n_val  # Update

    # shuffle
    random.seed(seed)
    random.shuffle(audio_files)

    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train+n_val]  # New line
    test_files = audio_files[n_train+n_val:]  # Update

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")  # New line
    print(f"Test files: {len(test_files)}")
    continue_ = input("Continue [yn]? ") or "n"

    if continue_ != "y":
        return

    split_dirs = {}
    for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):  # Update
        split_dir = Path(out_dir) / f"{audio_folder.name}"/ f"{split}"
        split_dirs[split] = split_dir
        for file in tqdm(files):
            out_file = (
                split_dir / Path(file).relative_to(folder)
            )
            out_file.parent.mkdir(exist_ok=True, parents=True)
            os.symlink(file, out_file)

        # save split as json
        with open(Path(audio_folder) / f"{split}.json", "w") as f:
            json.dump([str(f) for f in files], f)

    print(f"Done!")
    for split, split_dir in split_dirs.items():
        print(f"split {split} is located at \n{split_dir}\n\n")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        train_test_val_split()  # Update function name