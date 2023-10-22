from pathlib import Path
import pandas as pd
import yaml

import argbind

@argbind.bind(without_prefix=True)
def main(
    spec: str = "conf/salad_bowl/salad/salad-spec.yml", 
    prosound_dac_root: str = "data/codes-mono-win=10/prosound",
    output_file: str = "conf/salad_bowl/salad-data.csv"
):
    # load the spec
    # it's a dict, the keys are the class names
    # the vals are folders that we can glob for files
    with open(spec, "r") as f:
        spec = yaml.safe_load(f)

    print(f"Loaded spec with {len(spec)} classes")

    # collect metadata
    metadata = []
    for clsname, subpaths in spec.items():
        print(f"Collecting metadata for class {clsname}")
        for subpath in subpaths:
            subpath = Path(subpath)
            cross_split_files = []
            for _split in ["train", "val", "test"]:
                dacpath = Path(prosound_dac_root) / _split / subpath
                files = list(dacpath.glob("**/*.dac"))
                cross_split_files.extend([(_f, _split) for _f in files])
                print(f"Found {len(cross_split_files)} files in {dacpath}")

            for pair in cross_split_files:
                file, split = pair
                metadata.append({
                    "dac_path": file,
                    "label": clsname,
                    "split": split
                })
                
    metadata = pd.DataFrame(metadata)

    metadata.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()