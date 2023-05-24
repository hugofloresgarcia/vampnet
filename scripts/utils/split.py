from pathlib import Path
import random
import shutil

import argbind

from audiotools.core import util


@argbind.bind(without_prefix=True)
def train_test_split(
    audio_folder: str = ".", 
    test_size: float = 0.2,
    seed: int = 42,
):
    audio_files = util.find_audio(audio_folder)
    
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
        for file in files:
            out_file = Path(file).parent / split / Path(file).name
            out_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(file, out_file)
    

    
if __name__ == "__main__":
    args  = argbind.parse_args()

    with argbind.scope(args):
        train_test_split()