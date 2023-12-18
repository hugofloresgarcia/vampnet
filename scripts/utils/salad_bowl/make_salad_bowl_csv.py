from pathlib import Path
import pandas as pd
import yaml

import argbind

@argbind.bind(without_prefix=True)
def main(
    spec: str = "conf/salad_bowl/salad/salad-spec.yml", 
    prosound_metadata: str = "data/metadata/prosound_redacted.csv",
    output_file: str = "data/metadata/salad_bowl.csv"
):
    # load the spec
    # it's a dict, the keys are the class names
    # the vals are folders that we can glob for files
    with open(spec, "r") as f:
        spec = yaml.safe_load(f)

    print(f"Loaded spec with {len(spec)} classes")

    # load the prosound metadata
    prosound_metadata = pd.read_csv(prosound_metadata)

    # make a subset of the dataframe that only has the files we want
    # we can do this by iterating through each row's dac_path and seeing if 
    # it's a child of any of the folders in the spec
    # if it is, we keep in the subset, otherwise we drop it
    subset = []
    for _, row in prosound_metadata.iterrows():
        for class_name, folders in spec.items():
            for folder in folders:
                if folder in row["dac_path"]:
                    subset.append(row)
                    break
    
    print(f"Subset has {len(subset)} rows")

    # now we can make the metadata for the salad bowl and save it
    salad_bowl_metadata = pd.DataFrame(subset)
    salad_bowl_metadata.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()