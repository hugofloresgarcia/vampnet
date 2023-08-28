from pathlib import Path
from typing import Optional
import subprocess
import concurrent.futures
import argbind
import tqdm

def convert_mp3_to_wav(src, tgt, mp3_path: Path):
    wav_path = tgt / (mp3_path.relative_to(src).with_suffix(".wav"))
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists():
        print(f"Skipping {mp3_path} as {wav_path} already exists")
        return 0
    else:
        # print(f"Converting {mp3_path} to {wav_path}")
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(mp3_path), str(wav_path)],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {mp3_path} to {wav_path}")
            print(e.stderr)
            return 1  # Return 1 to indicate an error occurred
    return 0

@argbind.bind(without_prefix=True)
def mp3towav(src: str = None, tgt: str = None):
    src = Path(src)
    tgt = Path(tgt)

    tgt.mkdir(parents=True, exist_ok=True)
    
    error_count = 0  # Initialize error counter

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for mp3_path in tqdm.tqdm(src.glob("**/*.mp3")):
            futures.append(executor.submit(convert_mp3_to_wav, src, tgt, mp3_path))

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            error_count += future.result()

    # Print the count of files that raised a subprocess error
    if error_count > 0:
        print(f"\n{error_count} files failed to convert.")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        mp3towav()
