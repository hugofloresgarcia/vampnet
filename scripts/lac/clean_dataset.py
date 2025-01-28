import argbind
import audiotools as at
import tqdm
import shutil
from pathlib import Path
import sys

# Removes corrupt files from dataset

@argbind.bind("train", "val")
def build_dataset(
    folders: dict = None,
    trash_dir: str = "/data/trash"
):
    trash_dir = Path(trash_dir)
    trash_dir.mkdir(parents=True, exist_ok=True)
    for k, v in folders.items():
        loader = at.datasets.AudioLoader(sources=v)
        pbar_a = tqdm.tqdm(loader.audio_lists, desc=k)
        for i, item_list in enumerate(pbar_a):
            pbar_b = tqdm.tqdm(item_list, desc=loader.sources[i])
            for item in pbar_b:
                try:
                    at.AudioSignal(item['path'], duration=0.5)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    path = Path(item['path'])
                    trash_path = trash_dir / "/".join(path.parts[2:])
                    trash_path.parent.mkdir(exist_ok=True, parents=True)
                    print(f"Error loading {item['path']}, moving file to {str(trash_path)}.")
                    shutil.move(path, trash_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    for scope in ["train", "val"]:
        with argbind.scope(args, scope):
            build_dataset()

