import vampnet
import duckdb
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

def conn(read_only=True) -> duckdb.DuckDBPyConnection:
    if not hasattr(vampnet, '_conn'):
        # make sure that vampnet.DB's parent directory exists
        Path(vampnet.DB).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(vampnet.DB, read_only=read_only)
        vampnet._conn = conn    
    
    return vampnet._conn

@dataclass
class Dataset:
    name: str
    root: str
def create_dataset_table(conn):
    # create a table for datasets
    conn.sql(
        """
        CREATE SEQUENCE seq_dataset_id START 1
        """
    )
    conn.sql(
        """
        CREATE TABLE dataset (
            id INTEGER NOT NULL PRIMARY KEY DEFAULT nextval('seq_dataset_id'),
            name STRING NOT NULL UNIQUE,
            root STRING,
            UNIQUE (name, root)
        )
        """
    )

def insert_dataset(conn, dataset: Dataset):
    return conn.sql(
        f"""
        INSERT INTO dataset (
            name, root
        ) VALUES (
            '{dataset.name}', '{dataset.root}'
        )
        RETURNING id
        """
    ).fetchone()[0]

def get_dataset(conn, name: str) -> Dataset:
    return conn.execute(f"""
        SELECT id, root
        FROM dataset
        WHERE name = '{name}'
    """).fetchone()

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
def create_audio_file_table(conn):
    conn.sql(
        """
        CREATE SEQUENCE seq_audio_file_id START 1
        """
    )
    conn.sql(
        """
        CREATE TABLE audio_file (
            id INTEGER NOT NULL PRIMARY KEY DEFAULT nextval('seq_audio_file_id'),
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

def _denull(d):
    return d
    # # replace none with "null" string
    # for k, _v in d.__dict__.items():
    #     if _v is None:
    #         d.__dict__[k] = None
    # return d

def insert_audio_file(conn, audio_file: AudioFile):
    _denull(audio_file)
    # return conn.sql(
    #     f"""
    #     INSERT INTO audio_file  BY POSITION (
    #         path, dataset_id, num_frames, sample_rate, num_channels, bit_depth, encoding
    #     ) VALUES (
    #         '{audio_file.path}', {audio_file.dataset_id}, {audio_file.num_frames}, 
    #         {audio_file.sample_rate}, {audio_file.num_channels}, {audio_file.bit_depth}, 
    #         '{audio_file.encoding}'
    #     )
    #     RETURNING id
    #     """
    # ).fetchone()[0]
    # use ? instead of f-string
    return conn.execute(
        """
        INSERT INTO audio_file  BY POSITION (
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
    conn.insert("audio_file", df)


def get_audio_file_table(conn, dataset_id: int) -> pd.DataFrame:
    return conn.execute(f"""
        SELECT *
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).df()

# a table for control signals
@dataclass
class ControlSignal:
    path: str
    audio_file_id: int
    name: str
    hop_size: int
    num_frames: int
    num_channels: int
def create_ctrl_sig_table(conn):
    conn.sql(
        """
        CREATE SEQUENCE seq_ctrl_sig_id START 1
        """
    )
    conn.sql(
        """
        CREATE TABLE ctrl_sig (
            id INTEGER NOT NULL PRIMARY KEY DEFAULT nextval('seq_ctrl_sig_id'),
            path STRING NOT NULL UNIQUE,
            audio_file_id INTEGER NOT NULL,
            name STRING NOT NULL,
            hop_size INTEGER NOT NULL,
            num_frames INTEGER NOT NULL,
            num_channels INTEGER NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, name)
        )
        """
    )

def insert_ctrl_sig(conn, ctrl_sig: ControlSignal):
    _denull(ctrl_sig)
    # return conn.sql(
    #     f"""
    #     INSERT INTO ctrl_sig BY POSITION (
    #         path, audio_file_id, name, hop_size, num_frames, num_channels
    #     ) VALUES (
    #         '{ctrl_sig.path}', {ctrl_sig.audio_file_id}, '{ctrl_sig.name}', 
    #         {ctrl_sig.hop_size}, {ctrl_sig.num_frames}, {ctrl_sig.num_channels}
    #     )
    #     RETURNING id
    #     """
    # ).fetchone()[0]

    # use ? instead of f-string
    return conn.execute(
        """
        INSERT INTO ctrl_sig BY POSITION (
            path, audio_file_id, name, hop_size, num_frames, num_channels
        ) VALUES (
            ?, ?, ?, ?, ?, ?
        )
        RETURNING id
        """,
        (ctrl_sig.path, ctrl_sig.audio_file_id, ctrl_sig.name,
        ctrl_sig.hop_size, ctrl_sig.num_frames, ctrl_sig.num_channels)
    ).fetchone()[0]

# a table for train/test splits
@dataclass
class Split:
    audio_file_id: int
    split: str

def create_split_table(conn):
    conn.sql(
        """
        CREATE SEQUENCE seq_split_id START 1
        """
    )
    conn.sql(
        """
        CREATE TABLE split (
            id INTEGER NOT NULL PRIMARY KEY DEFAULT nextval('seq_split_id'),
            audio_file_id INTEGER NOT NULL,
            split STRING NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, split)
        )
        """
    )

def insert_split(conn, split: Split):
    # breakpoint()
    conn.sql(
        f"""
        INSERT INTO split ( audio_file_id, split ) 
        VALUES ({split.audio_file_id}, '{split.split}')
        """
    )



def init():
    from vampnet.db import (
        create_dataset_table, create_audio_file_table,
        create_ctrl_sig_table, create_split_table
    )
    for fn in [
        create_dataset_table,
        create_audio_file_table,
        create_ctrl_sig_table,
        create_split_table
    ]:
        conn = vampnet.db.conn(read_only=False)
        try: 
            print(f"running {fn.__name__}")
            fn(conn)
        except duckdb.Error as e:
            print(f"error: {e}")

    print("done! :)")
