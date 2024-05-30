"""
train-val-test split on a dataset 
specifically, we want to create splits along 
audio_file_id
"""
from pathlib import Path
import audiotools as at

import argbind
import tqdm
import duckdb
import pandas as pd

import vampnet

def partition_dataset(
    dataset: str = vampnet.DATASET,
    train: float = vampnet.TRAIN_PROPORTION,
    val: float = vampnet.VAL_PROPORTION,
    test: float = vampnet.TEST_PROPORTION 
):
    assert dataset is not None
    # make sure our proportions sum to 1
    assert train + val + test == 1.0

    # connect to our datasets table
    conn = vampnet.db.conn(read_only=False)
    # begin a transaction
    conn.sql("BEGIN TRANSACTION")

    # get the dataset id and root
    dataset_id, root = vampnet.db.get_dataset(conn, dataset)
    print(f"Found dataset {dataset} at {root}")

    # get the audio files and their ids
    audio_file_ids = conn.execute(f"""
        SELECT id 
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).fetchall()
    print(f"Found {len(audio_file_ids)} audio files")

    # shuffle the audio files
    import random
    random.shuffle(audio_file_ids)

    # calculate the number of audio files in each split
    n_train = int(train * len(audio_file_ids))
    n_val = int(val * len(audio_file_ids))
    n_test = len(audio_file_ids) - n_train - n_val

    # insert the splits into the partition table
    for i, audio_file_id in enumerate(audio_file_ids):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'

        split = vampnet.db.Split(
            audio_file_id=audio_file_id[0],
            split=split
        )

        vampnet.db.insert_split(conn, split)


    conn.sql("COMMIT")
    print("done! :)")


if __name__ == "__main__":
    # args = argbind.parse_args()
    # partition_dataset = argbind.bind(without_prefix=True)(partition_dataset)

    # with argbind.scope(args):
    partition_dataset()