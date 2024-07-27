from pathlib import Path
import audiotools as at

import argbind
import tqdm
import sqlite3
import torchaudio
import pandas as pd

import vampnet

def create_dataset(
    audio_folder: str = vampnet.AUDIO_FOLDER,
    dataset_name: str = vampnet.DATASET
):
    """creates a new sqlite3 dataset, and populates it with audio files from a folder.
    """
    # connect to our datasets table
    conn = sqlite3.connect(vampnet.DB)
    # begin a transaction
    # conn.execute("BEGIN TRANSACTION")

    # create a new dataset
    if not vampnet.db.dataset_exists(conn, dataset_name):
        vampnet.db.insert_dataset(conn, vampnet.db.Dataset(name=dataset_name, root=audio_folder))


    dataset_id = conn.execute(f"SELECT id FROM dataset WHERE name = '{dataset_name}'").fetchone()[0]

    # create a new table for our audio files
    print(f"looking for audio files in {audio_folder}")
    audio_files = at.util.find_audio(
        Path(audio_folder), ext=[".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"]
    )
    print(f"Found {len(audio_files)} audio files")

    # for file in tqdm.tqdm(audio_files):

    # num_failed = 0
    def process_file(file):
        try: 
            info = torchaudio.info(file, backend="ffmpeg")
            path = str(Path(file).absolute().relative_to(Path(audio_folder).absolute()))
            print(f"{path}")
            if info.num_channels > vampnet.AUDIO_LOOKUP_MAX_AUDIO_CHANNELS:
                raise ValueError(f"skipping {file} because it has {info.num_channels} channels!!!!")

            if not vampnet.db.audio_file_exists(conn, path, dataset_id):
                af = vampnet.db.AudioFile(
                    path=path,
                    dataset_id=dataset_id,
                    num_frames=info.num_frames,
                    sample_rate=info.sample_rate,
                    num_channels=info.num_channels,
                    bit_depth=info.bits_per_sample,
                    encoding=info.encoding
                )
                vampnet.db.insert_audio_file(conn, af)
                conn.commit()
            else:
                print(f"Skipping {file} because it already exists in the db")
                return None
            
        except Exception as e:
            # global num_failed
            print(f"Could not process {file}: {e}")
            # num_failed += 1
            return None
        
        return af


    # from vampnet.util import parallelize
    # afs = parallelize(process_file, audio_files, parallel="thread_map")
    from concurrent.futures import ThreadPoolExecutor
    import tqdm
    with ThreadPoolExecutor() as executor:
        afs = list(tqdm.tqdm(map(process_file, audio_files), total=len(audio_files)))

    # now, we can write to the db 
    # for af in afs:
        # if af is not None:
            # vampnet.db.insert_audio_file(conn, af)

        
    # ask if we should commit
    print(f"Processed {len(audio_files)} audio files")
    # print(f"of which {num_failed} failed")
    print("done! committing to the db.")
    # conn.execute("COMMIT")

if __name__ == "__main__":
    import yapecs

    parser = yapecs.ArgumentParser()

    parser.add_argument("--audio_folder", type=str, default=vampnet.AUDIO_FOLDER, help="folder containing audio files")
    parser.add_argument("--dataset_name", type=str, default=vampnet.DATASET, help="name of the dataset")

    args = parser.parse_args()

    create_dataset(
        audio_folder=args.audio_folder,
        dataset_name=args.dataset_name
    )
    

