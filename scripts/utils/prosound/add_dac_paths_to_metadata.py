from pathlib import Path
import json

import pandas as pd
from ast import literal_eval

import argbind


@argbind.bind(without_prefix=True)
def main(
    path_to_metadata: str = None, 
    audio_root: str = None,
    dac_root: str = None, 
    output_path: str = "./data/metadata/metadata-with-family.csv"
):
    print(f"loading metadata from {path_to_metadata}")
    metadata = pd.read_csv(path_to_metadata)
    print(f"found {len(metadata)} rows in metadata")
    # metadata columns: duration,sample_rate,filename,name,num_channels,tags

    audio_path = Path(audio_root)
    dac_path = Path(dac_root)

    metadata["dac_path"] = metadata["filename"].apply(lambda x: str(dac_path / str(Path(x).relative_to(audio_root)).replace(".wav", ".dac")))

    # check how many of these exist
    metadata["dac_exists"] = metadata["dac_path"].apply(lambda x: Path(x).exists())

    print(f"{metadata['dac_exists'].sum()} DAC files exist out of {len(metadata)}")

    metadata.to_csv(output_path, index=False)





if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()