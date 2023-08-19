from pathlib import Path

from audiotools import AudioSignal
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

paths = (
    "/media/CHONK/hugo/spotdl/audio-train",
    "/media/CHONK/hugo/spotdl/audio-test",
    "/media/CHONK/hugo/spotdl/audio-val",
)


# num_deleted = 0
def test_file(file):
    try:
        AudioSignal(str(file), offset=0, duration=0.1)
    except:
        print(f"Failed to load {file}")
        file.unlink()
        # num_deleted += 1
        print(f"Deleted {file}")


for path in paths:
    # glob all files in path
    files = list(Path(path).glob("**/*.mp3"))
    process_map(test_file, files, max_workers=8, chunksize=20)
    # for file in tqdm(files):
    # test_file(file)


# pritn(num_deleted)
