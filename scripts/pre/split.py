import pandas as pd
from pathlib import Path
import random
import shutil
import argbind
from tqdm import tqdm

@argbind.bind(without_prefix=True)
def train_test_val_split(
    input_csv: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    # 1. Backup the CSV
    backup_file = f"{input_csv}.backup"
    shutil.copy(input_csv, backup_file)
    print(f"Backup created at: {backup_file}")

    # 2. Read CSV and determine splits
    df = pd.read_csv(input_csv)
    print(f"Loaded metadata with {len(df)} rows")

    # Determine the indices for each split
    n_test = int(len(df) * test_size)
    n_val = int(len(df) * val_size)
    n_train = len(df) - n_test - n_val

    indices = list(range(len(df)))
    random.seed(seed)
    random.shuffle(indices)

    # Assign splits in the dataframe based on shuffled indices
    df['split'] = ""
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    df.loc[train_indices, 'split'] = "train"
    df.loc[val_indices, 'split'] = "val"
    df.loc[test_indices, 'split'] = "test"

    # 3. Save the updated dataframe
    df.to_csv(input_csv, index=False)
    print(f"Updated metadata saved to {input_csv}")

    print(f"Train files: {len(df[df['split'] == 'train'])}")
    print(f"Validation files: {len(df[df['split'] == 'val'])}")
    print(f"Test files: {len(df[df['split'] == 'test'])}")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        train_test_val_split()
