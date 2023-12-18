from pathlib import Path
import pandas as pd
import shutil
import argbind

# TODO: should make a library of function wrappers that 
# operates this csv data pattern:
#  before the operation, make a backup of the csv and load it
#  after the operation, save the csv
from scipy import stats
from vampnet.condition import YamnetConditioner, ConditionFeatures
from tqdm import tqdm

@argbind.bind(without_prefix=True)
def add_yamnet_labels(
    input_csv: Path = None, 
):
    # make a backup of the metadata
    shutil.copy(input_csv, f"{input_csv}.backup")

    df = pd.read_csv(input_csv)
    print(f"Loaded metadata with {len(df)} rows")

    yamnet = YamnetConditioner()

    # we're gonna add a label column to the dataframe
    
    # add an empty label column
    df['label_top1'] = ""
    df['label_top2'] = ""
    df['label_top3'] = ""
    df['label_top4'] = ""
    df['label_top5'] = ""

    # iterate over the rows
    pbar = tqdm(df.iterrows(), total=len(df))
    for i, row in pbar:
        # get the yamnet path
        yamnet_path = Path(row['yamnet_root']) /  Path(row['yamnet_path'])

        # load the yamnet features
        yamnet_features = ConditionFeatures.load(yamnet_path)

        # get the scores
        scores = yamnet_features.features['scores']

        class_names = yamnet.class_names
        # get the top 1, 2 and 3 labels
        # these are still time varying, so we have to take the majority vote
        indices = scores.argsort(axis=-1)
        top1 = indices[:, -1]
        top2 = indices[:, -2]
        top3 = indices[:, -3]
        top4 = indices[:, -4]
        top5 = indices[:, -5]

        # get the majority vote
        top1 = stats.mode(top1)[0]
        top2 = stats.mode(top2)[0]
        top3 = stats.mode(top3)[0]
        top4 = stats.mode(top4)[0]
        top5 = stats.mode(top5)[0]

        # add the label to the dataframe
        df.loc[i, 'label_top1'] = class_names[top1]
        df.loc[i, 'label_top2'] = class_names[top2]
        df.loc[i, 'label_top3'] = class_names[top3]
        df.loc[i, 'label_top4'] = class_names[top4]
        df.loc[i, 'label_top5'] = class_names[top5]

        pbar.set_description(f"Top 3 labels: {class_names[top1]}, {class_names[top2]}, {class_names[top3]}, {class_names[top4]}, {class_names[top5]}")

    # are we done? then, save the dataframe
    df.to_csv(input_csv, index=False)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        add_yamnet_labels()







        




    