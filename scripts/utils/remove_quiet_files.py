# removes files with loudness below 24db

from pathlib import Path 
import shutil
import audiotools as at
import argbind

@argbind.bind(without_prefix=True)
def remove_quiet_files(
    src_dir: Path = None,
    dest_dir: Path = None,
    min_loudness: float = -30,
):
    # copy src to dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    
    audio_files = at.util.find_audio(dest_dir)
    for audio_file in audio_files:
        sig = at.AudioSignal(audio_file)
        if sig.loudness() < min_loudness:
            audio_file.unlink()
            print(f"removed {audio_file}")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        remove_quiet_files()