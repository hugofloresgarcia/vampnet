from pathlib import Path

import pandas as pd

import argbind
import audiotools as at
import tqdm


# todo: add split
@argbind.bind(without_prefix=True)
def main(
    audio_root: str = None,
    dac_root: str = None, 
    output_path: str = "disco-metadata.csv"
):
    audio_path = Path(audio_root)
    dac_path = Path(dac_root)

    # find all the audio files
    audio_files = at.util.find_audio(audio_path)
    print(f"found {len(audio_files)} audio files")

    def get_split(filename):
        split =  str(filename).split("/")[2]
        assert split in ("train", "val", "test")
        return split

    metadata = []
    for filename in tqdm.tqdm(audio_files):
        metadata.append({
            "filename": filename, 
            "family": "Music", 
            "dac_path": str(dac_path / str(Path(filename).relative_to(audio_path)).replace(".mp3", ".dac")), 
            "split": get_split(filename)
        })
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()