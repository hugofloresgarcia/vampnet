from pathlib import Path
import audiotools as at

import argbind
import tqdm
import duckdb
import pandas as pd

import vampnet

RAISE = True

def preprocess(
    dataset: str = None
):
    """
    encodes the audio files with the codec model
    and populates them into the database. 
    """
    assert dataset is not None

    # connect to our datasets table
    conn = vampnet.db.conn(read_only=False)
    # begin a transatcion
    conn.sql("BEGIN TRANSACTION")

    # get the dataset id and root
    dataset_id, root = conn.execute(f"""
        SELECT id, root 
        FROM dataset 
        WHERE name = '{dataset}'
    """).fetchone()
    print(f"Found dataset {dataset} at {root}")

    # get the audio files and their ids
    audio_files = conn.execute(f"""
        SELECT id, path 
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).fetchall()
    print(f"Found {len(audio_files)} audio files")

    for Ctrl in vampnet.controls.load_control_signal_extractors():
        print(f"processing control signal {Ctrl.name}")

        # encode the audio files
        num_failed = 0
        for audio_id, path in tqdm.tqdm(audio_files):
            try: 
                sig = at.AudioSignal(Path(root) / path)
                ctrlsig = Ctrl.from_signal(sig)
                
                out_path = Path(vampnet.CACHE_PATH) / dataset / Path(path).with_suffix(Ctrl.ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ctrlsig.save(out_path)

                # add to our ctrl sig table
                ctrlsig_file = vampnet.db.ControlSignal(
                    path=str(out_path.absolute().relative_to(Path(vampnet.CACHE_PATH / dataset).absolute())),
                    audio_file_id=audio_id, 
                    name=Ctrl.name,
                    hop_size=ctrlsig.hop_size,
                    num_frames=ctrlsig.num_frames,
                    num_channels=ctrlsig.num_channels
                )
                try:
                    vampnet.db.insert_ctrl_sig(conn, ctrlsig_file)
                except:
                    print(f"Could not insert {out_path} into control signal table")
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
    conn.sql("COMMIT")



if __name__ == "__main__":
    import yapecs
    parser = yapecs.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None, help="dataset to preprocess")

    args = parser.parse_args()
    preprocess(**vars(args))