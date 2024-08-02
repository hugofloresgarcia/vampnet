import vampnet
import sqlite3
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

def conn() -> sqlite3.Connection:
    if not hasattr(vampnet, '_conn'):
        # make sure that vampnet.DB's parent directory exists
        Path(vampnet.DB).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(vampnet.DB)
        vampnet.db._conn = conn    

        # print all the tables in the database
        print(f"loaded database from {vampnet.DB}")
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print("tables in the database:")
        print(cur.fetchall())
    
    return vampnet.db._conn


def cursor():
    conn = vampnet.db.conn()
    return conn.cursor()

@dataclass
class Dataset:
    name: str
    root: str
def create_dataset_table(cur):
    cur.execute(
        """
        CREATE TABLE dataset (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name STRING NOT NULL UNIQUE,
            root STRING,
            UNIQUE (name, root)
        )
        """
    )

def insert_dataset(cur, dataset: Dataset):
    return cur.execute(
        f"""
        INSERT INTO dataset (
            name, root
        ) VALUES (
            '{dataset.name}', '{dataset.root}'
        )
        RETURNING id
        """
    ).fetchone()[0]

def get_dataset(cur, name: str) -> Dataset:
    return cur.execute(f"""
        SELECT id, root
        FROM dataset
        WHERE name = '{name}'
    """).fetchone()

def dataset_exists(cur, name: str) -> bool:
    return cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM dataset
            WHERE name = '{name}'
        )
    """).fetchone()[0]

# create a table for audio files
@dataclass
class AudioFile:
    dataset_id: int
    path: Optional[str] = None
    num_frames: Optional[int] = None
    sample_rate: Optional[int] = None
    num_channels: Optional[int] = None
    bit_depth: Optional[int] = None
    encoding: Optional[str] = None
def create_audio_file_table(cur):
    cur.execute(
        """
        CREATE TABLE audio_file (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            path STRING,
            dataset_id INTEGER NOT NULL,
            num_frames INTEGER,
            sample_rate INTEGER,
            num_channels INTEGER,
            bit_depth INTEGER ,
            encoding STRING,
            FOREIGN KEY (dataset_id) REFERENCES dataset(id),
            UNIQUE (path, dataset_id)
        )
        """
    )


def insert_audio_file(cur, audio_file: AudioFile):
    return cur.execute(
        """
        INSERT INTO audio_file (
            path, dataset_id, num_frames, sample_rate, num_channels, bit_depth, encoding
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?
        )
        RETURNING id
        """,
        (audio_file.path, audio_file.dataset_id, audio_file.num_frames, 
        audio_file.sample_rate, audio_file.num_channels, audio_file.bit_depth, 
        audio_file.encoding)
    ).fetchone()[0]

    # convert to a dataframe and insert with dataframe
    df = pd.DataFrame([audio_file.__dict__])
    cur.insert("audio_file", df)


def get_audio_file_table(cur, dataset_id: int) -> pd.DataFrame:
    return cur.execute(f"""
        SELECT *
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).df()

def audio_file_exists(cur, path: str, dataset_id: int) -> bool:
    return cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM audio_file
            WHERE path = '{path}'
            AND dataset_id = {dataset_id}
        )
    """).fetchone()[0]

# a table for control signals
@dataclass
class ControlSignal:
    path: str
    audio_file_id: int
    name: str
    sample_rate: int
    hop_size: int
    num_frames: int
    num_channels: int
def create_ctrl_sig_table(cur):
    cur.execute(
        """
        CREATE TABLE ctrl_sig (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            path STRING NOT NULL UNIQUE,
            audio_file_id INTEGER NOT NULL,
            name STRING NOT NULL,
            sample_rate INTEGER NOT NULL,
            hop_size INTEGER NOT NULL,
            num_frames INTEGER NOT NULL,
            num_channels INTEGER NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id)
        )
        """
    )

def insert_ctrl_sig(cur, ctrl_sig: ControlSignal):
    return cur.execute(
        """
        INSERT INTO ctrl_sig (
            path, audio_file_id, name, sample_rate, hop_size, num_frames, num_channels
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?
        )
        RETURNING id
        """,
        (ctrl_sig.path, ctrl_sig.audio_file_id, ctrl_sig.name, ctrl_sig.sample_rate,
        ctrl_sig.hop_size, ctrl_sig.num_frames, ctrl_sig.num_channels)
    ).fetchone()[0]

def ctrl_sig_exists(cur, path: str, audio_file_id: int) -> bool:
    # print(f'looking for ctrl_sig {path} for audio_file_id {audio_file_id}')
    out =  cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM ctrl_sig
            WHERE path = '{path}'
            AND audio_file_id = {audio_file_id}
        )
    """).fetchone()[0]
    # print(f'ctrl_sig exists: {out}')
    return out

# a table for train/test splits
@dataclass
class Split:
    audio_file_id: int
    split: str

def create_split_table(cur):
    cur.execute(
        """
        CREATE TABLE split (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            audio_file_id INTEGER NOT NULL,
            split STRING NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, split)
        )
        """
    )

def insert_split(cur, split: Split):
    # breakpoint()
    cur.execute(
        f"""
        INSERT INTO split ( audio_file_id, split ) 
        VALUES ({split.audio_file_id}, '{split.split}')
        """
    )


@dataclass
class Caption:
    text: str
    audio_file_id: int

def create_caption_table(cur):
    cur.execute(
        """
        CREATE TABLE caption (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            text STRING NOT NULL,
            audio_file_id INTEGER NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id)
        )
        """
    )

def insert_caption(cur, caption: Caption):
    cur.execute(
        """
        INSERT INTO caption ( text, audio_file_id ) 
        VALUES (?, ?)
        """,
        (caption.text, caption.audio_file_id)
    )


def init():
    for fn in [
        create_dataset_table,
        create_audio_file_table,
        create_ctrl_sig_table,
        create_split_table, 
        create_caption_table
    ]:
        cur = cursor()
        try: 
            print(f"running {fn.__name__}")
            fn(cur)
        except Exception as e:
            print(f"error: {e}")

    print("done! :)")
