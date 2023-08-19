import concurrent.futures
from pathlib import Path
import shutil
from tqdm import tqdm


def move_file(file):
    """Move file to a new directory based on the file name"""
    new_dirs = [file.stem[i:i+3] for i in range(0, 10, 3)]
    new_path = file.parent / Path(*new_dirs)
    new_path.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file), str(new_path))


def glob_and_move_files(directory):
    """Glob files in directory and move files concurrently"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        p = Path(directory)
        files = list(tqdm(p.glob("**/spotify:track:*.mp3")))

        with tqdm(total=len(files), desc="Moving Files", ncols=80) as pbar:
            list(tqdm(executor.map(move_file, files), total=len(files)))
            pbar.update(len(files))


if __name__ == "__main__":
    dir_to_clean = "/media/CHONK/hugo/spotdl/audio-train"
    glob_and_move_files(dir_to_clean)
