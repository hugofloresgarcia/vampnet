from pathlib import Path
import audiotools as at
import math

import argbind
import tqdm
import sqlite3
import pandas as pd
import torch.nn as nn

import vampnet
from vampnet.signal import Signal

RAISE = False

def preprocess(
    dataset: str = vampnet.DATASET
):
    """
    encodes the audio files with the codec model
    and populates them into the database. 
    """
    assert dataset is not None

    # connect to our datasets table
    conn = sqlite3.connect(vampnet.DB_FILE)
    cur = vampnet.db.cursor()

    # get the dataset id and root
    dataset_id, root = cur.execute(f"""
        SELECT id, root 
        FROM dataset 
        WHERE name = '{dataset}'
    """).fetchone()
    print(f"Found dataset {dataset} at {root}")

    # get the audio files and their ids
    audio_files = cur.execute(f"""
        SELECT id, path 
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).fetchall()
    print(f"Found {len(audio_files)} audio files")

    # shuffle audio files
    audio_files = [(audio_id, path) for audio_id, path in audio_files]
    import random
    random.shuffle(audio_files)

    num_failed = 0
    for Ctrl in vampnet.controls.load_control_signal_extractors():
        print(f"processing control signal {Ctrl.name}")

        # encode the audio files
        for audio_id, path in tqdm.tqdm(audio_files):
            try: 
                out_path = Path(vampnet.CACHE_PATH) / dataset / Path(path).with_suffix(Ctrl.ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                dbpath = str(out_path.absolute().relative_to(Path(vampnet.CACHE_PATH / dataset).absolute()))

                if out_path.exists() and vampnet.db.ctrl_sig_exists(cur, dbpath, audio_file_id=audio_id):
                    # print(f"Control signal {Ctrl.name} already exists for {path}")
                    continue

                sig = Signal(Path(root) / path)
                sig.to(vampnet.DEVICE)

                # preprocess the sig for the codec
                length = sig.samples.shape[-1]
                right_pad = math.ceil(length / vampnet.HOP_SIZE) * vampnet.HOP_SIZE - length
                sig.samples = nn.functional.pad(sig.samples, (0, right_pad))

                if out_path.exists():
                    print(f'reloading from file!')
                    # load the ctrlsig
                    ctrlsig = Ctrl.load(out_path)
                else:
                    ctrlsig = Ctrl.from_signal(sig.detach().cpu())
                
                ctrlsig.save(out_path)

                # add to our ctrl sig table
                ctrlsig_file = vampnet.db.ControlSignal(
                    path=dbpath,
                    audio_file_id=audio_id, 
                    name=Ctrl.name,
                    sample_rate=sig.sample_rate,
                    hop_size=ctrlsig.hop_size,
                    num_frames=ctrlsig.num_frames,
                    num_channels=ctrlsig.num_channels
                )
                try:
                    cur.execute("BEGIN")
                    vampnet.db.insert_ctrl_sig(cur, ctrlsig_file)
                    cur.execute("COMMIT")
                except Exception as e:
                    print(f"Could not insert {out_path} into control signal table")
                    print(e)
                    cur.execute("ROLLBACK")
                    num_failed += 1
            except Exception as e:
                if RAISE:
                    raise e
                print(f"Could not process {path}: {e}")
                num_failed += 1

    # ask if we should commit
    print(f"Processed {len(audio_files)} audio files")
    print(f"of which {num_failed} failed")
    
    print("committing changes to the db.")
    cur.execute("COMMIT")



if __name__ == "__main__":
    import yapecs
    parser = yapecs.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=vampnet.DATASET, help="dataset to preprocess")

    args = parser.parse_args()
    preprocess(**vars(args))