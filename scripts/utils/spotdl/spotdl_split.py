from pathlib import Path
import json

import pandas as pd
import shutil
import tqdm

splits = ("train", "val", "test")
split_dir = "/media/CHONK/hugo/spotdl/splits"
split_filepaths = [Path(split_dir) / f"{split}.json" for split in splits]

metadata_path = "./data/metadata/spotdl.csv"

# backup the metadata
shutil.copy(metadata_path, metadata_path.replace(".csv", ".csv.backup"))

# load the split filepaths, should be lists of spotify ids
split_ids = {
    split: json.load(open(split_filepath)) for split, split_filepath in zip(splits, split_filepaths)
}
print(f"Loaded {len(split_ids['train'])} train ids, {len(split_ids['val'])} val ids, and {len(split_ids['test'])} test ids")

# load the metadata
metadata = pd.read_csv(metadata_path)
print(f"Loaded {len(metadata)} rows of metadata")

# add an id column to the metadata by taking the audio_path, keeping the stem, and removing the extension
metadata["id"] = metadata["audio_path"].apply(lambda x: Path(x).stem)
print(f"Added id column to metadata")

# Initialize split column with 'unknown' to catch any ids not found
metadata["split"] = "unknown"

# Vectorized approach to assigning splits
for split, ids in split_ids.items():
    metadata.loc[metadata["id"].isin(ids), "split"] = split

# Save the metadata
metadata.to_csv(metadata_path, index=False)

# Output some information about the splits
print(f"Added split column to metadata")
print(f"Split counts: {metadata['split'].value_counts()}")
print(f"Split percentages: {metadata['split'].value_counts(normalize=True)}")


