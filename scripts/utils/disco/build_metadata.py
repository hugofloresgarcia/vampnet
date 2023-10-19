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
    dac_path = Path(dac_root)

    dac_paths = list(dac_path.glob("**/*.dac"))
    print(f"found {len(dac_paths)} dac files")

    def get_split(filename):
        split =  str(filename).split("/")[3]
        assert split in ("train", "val", "test")
        return split

    metadata = []
    for filename in tqdm.tqdm(dac_paths):
        metadata.append({
            # "filename": filename.with_suffix(".mp3"), 
            "family": "Music", 
            "dac_path": filename,
            "split": get_split(filename)
        })
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()