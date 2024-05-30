from pathlib import Path
import audiotools as at

import argbind
import tqdm
import duckdb
import torchaudio
import pandas as pd

import vampnet

def create_dataset(
    audio_folder: str = None,
    dataset_name: str = None
):
    """creates a new duckdb dataset, and populates it with audio files from a folder.
    """
    assert audio_folder is not None
    assert dataset_name is not None
    
    # connect to our datasets table
    conn = duckdb.connect(vampnet.DB)
    # begin a transaction
    conn.sql("BEGIN TRANSACTION")

    # create a new dataset
    vampnet.db.insert_dataset(conn, vampnet.db.Dataset(name=dataset_name, root=audio_folder))
    dataset_id = conn.execute(f"SELECT id FROM dataset WHERE name = '{dataset_name}'").fetchone()[0]

    # create a new table for our audio files
    print(f"looking for audio files in {audio_folder}")
    audio_files = at.util.find_audio(
        Path(audio_folder), ext=[".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"]
    )
    print(f"Found {len(audio_files)} audio files")

    # for file in tqdm.tqdm(audio_files):

    num_failed = 0
    def process_file(file):
        try: 
            info = torchaudio.info(file, backend="ffmpeg")
            af = vampnet.db.AudioFile(
                path=str(Path(file).absolute().relative_to(Path(audio_folder).absolute())),
                dataset_id=dataset_id,
                num_frames=info.num_frames,
                sample_rate=info.sample_rate,
                num_channels=info.num_channels,
                bit_depth=info.bits_per_sample,
                encoding=info.encoding
            )
            
        except Exception as e:
            global num_failed
            print(f"Could not process {file}: {e}")
            num_failed += 1
        
        return af


    # from vampnet.util import parallelize
    # afs = parallelize(process_file, audio_files, parallel="thread_map")
    from concurrent.futures import ThreadPoolExecutor
    import tqdm
    # with ThreadPoolExecutor() as executor:
    afs = list(tqdm.tqdm(map(process_file, audio_files), total=len(audio_files)))

    # now, we can write to the db 
    for af in afs:
        if af is not None:
            vampnet.db.insert_audio_file(conn, af)

        
    # ask if we should commit
    print(f"Processed {len(audio_files)} audio files")
    print(f"of which {num_failed} failed")
    print("done! committing to the db.")
    conn.sql("COMMIT")

if __name__ == "__main__":
    import yapecs

    parser = yapecs.ArgumentParser()

    parser.add_argument("--audio_folder", type=str, default=None, help="folder containing audio files")
    parser.add_argument("--dataset_name", type=str, default=None, help="name of the dataset")

    args = parser.parse_args()

    create_dataset(
        audio_folder=args.audio_folder,
        dataset_name=args.dataset_name
    )
    

