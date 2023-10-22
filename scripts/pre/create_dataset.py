""""create a basic metadata.csv from a folder of audio files"""

from pathlib import Path
import pandas as pd
import audiotools as at

import argbind
import tqdm

@argbind.bind(without_prefix=True)
def main(
    audio_folder: str = None,
    output_file: str = None,
):
    print(f"looking for audio files in {audio_folder}")
    audio_files = at.util.find_audio(
        Path(audio_folder), ext=[".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"]
    )
    print(f"Found {len(audio_files)} audio files")
    
    metadata = []
    for file in tqdm.tqdm(audio_files):
        metadata.append({
            "audio_path": Path(file).absolute().relative_to(Path(audio_folder)),
            "audio_root": Path(audio_folder).absolute(),
        })
    
    metadata = pd.DataFrame(metadata)

    print(f"Saving metadata to {output_file}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()