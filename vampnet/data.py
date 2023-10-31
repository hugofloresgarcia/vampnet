from typing import List
from pathlib import Path
import torch
import random
import time
import yaml

import pandas as pd
from audiotools import util

from audiotools import AudioSignal
from dac.model.base import DACFile
from dac.utils import load_model as load_dac

def filter_by_classlist(metadata: pd.DataFrame, classlist: List[str], label_key: str):
    """
    Filters the metadata by the classlist and returns the filtered metadata.
    """
    assert label_key in metadata.columns, f"Class key {label_key} not in metadata columns {metadata.columns}"
    # check that all the classes in the classlist are in the metadata
    for _cls in classlist:
        tru_classlist = metadata[label_key].unique().tolist()
        assert _cls in tru_classlist, f"Class {_cls} not in metadata classlist {tru_classlist}"
    metadata = metadata[metadata[label_key].isin(classlist)]
    return metadata

def filter_by_split(metadata: pd.DataFrame, split: str):
    """
    Filters the metadata by the split and returns the filtered metadata.
    """
    assert split in ["train", "val", "test"], f"Split must be one of [train, val, test], got {split}"
    metadata = metadata[metadata['split'] == split]
    return metadata

def add_p_column(metadata):
    """
    add a p column referring to the probability of sampling that row
    """
    assert "p" not in metadata.columns, f"p column already exists in metadata, cannot overwrite :("
    metadata['p'] = 1 / len(metadata)
    return metadata

def apply_class_weights(metadata: pd.DataFrame, class_weights: dict, label_key: str):
    """
    Applies the class weights to the metadata and returns the metadata.
    """
    assert label_key in metadata.columns, f"Class key {label_key} not in metadata columns {metadata.columns}"
    assert class_weights.keys() == metadata[label_key].unique().tolist(), f"Class weights keys {class_weights.keys()} do not match metadata classlist {metadata[label_key].unique().tolist()}"
    metadata['p'] = metadata[label_key].apply(lambda x: class_weights[x])
    return metadata


def pivot_by_type(
    df: pd.DataFrame, 
    id_key: str, 
    type_key: str,
):
    # Identify columns other than 'id' and 'type' to pivot
    cols_to_pivot = [col for col in df.columns if col not in [id_key, type_key]]

    # Initialize with a dataframe that only has 'id'
    result = df[[id_key]].drop_duplicates().reset_index(drop=True)

    for col in cols_to_pivot:
        # Pivot the current column based on 'id' and 'type'
        pivot = df.pivot(index=id_key, columns=type_key, values=col)
        pivot.columns = [f"{col}_{type_val}" for type_val in pivot.columns]
        
        # Merge the pivoted column to the result
        result = result.merge(pivot, left_on=id_key, right_index=True)

    return result

# if our seq is too short, then we need to pad it
def pad_if_needed(codes, seq_len):
    pad_mask = torch.ones(1, seq_len).int()
    if codes.shape[-1] < seq_len:
        pad_len = seq_len - codes.shape[-1]
        codes = torch.nn.functional.pad(codes, (0, pad_len))
        pad_mask[:, -pad_len:] = 0
    return codes, pad_mask

class DACDataset(torch.utils.data.Dataset):

    def __init__(self, 
        metadata_csvs: List[str] = None,
        seq_len: int = 1024,
        split: str = None, 
        classlist: list = None,
        paired: bool = False,
        main_key: str = "dac",
        type_key: str = "type",
        id_key: str = "id",
        label_key: str = "label",
        class_weights: dict = None, 
        length: int = 1000000000
    ):
        assert metadata_csvs is not None, "Must provide metadata_csvs"

        self.seq_len = seq_len
        self.length = length
        
        # load the metadata csvs
        self.metadata = []
        for csv in metadata_csvs:
            self.metadata.append(pd.read_csv(csv))
        
        self.metadata = pd.concat(self.metadata)
        print(f"loaded metadata with {len(self.metadata)} rows")

        # filter by split
        if split is not None:
            self.metadata = filter_by_split(self.metadata, split)
        print(f"resolved split: {split}")

        # check dac_keys
        self.paired = paired
        if self.paired:
            assert id_key in self.metadata.columns, f"id_key {id_key} not in metadata columns {self.metadata.columns}"
            assert type_key in self.metadata.columns, f"type_key {type_key} not in metadata columns {self.metadata.columns}"

            # pivot the dataframe
            self.type_keys = self.metadata["type"].unique().tolist()
            self.metadata = pivot_by_type(self.metadata, id_key, type_key)
            assert main_key in self.type_keys, f"main_key {main_key} not in type_keys {self.type_keys}"
            self.main_key = main_key
        else:
            # add dummy type column
            self.metadata[type_key] = main_key
            self.type_keys = [main_key]

        # add p column
        self.metadata = add_p_column(self.metadata)

        self.label_key = None
        if self.label_key is not None:
            assert label_key in self.metadata.columns, f"Class key {label_key} not in metadata columns {self.metadata.columns}"
            self.label_key = label_key
            
            # resolve classlist
            self.classlist = classlist if classlist is not None else self.metadata[label_key].unique().tolist()
            self.metadata = filter_by_classlist(self.metadata, classlist, label_key)
            print(f"resolved classlist: {self.classlist}")
            print(f'metadata now has {len(self.metadata)} rows')

            # resolve class weights
            self.class_weights = class_weights
            if self.class_weights is not None:
                self.metadata = apply_class_weights(self.metadata, class_weights, label_key)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx, attempt=0):
        util.seed(idx)
        data = {type_key: {} for type_key in self.type_keys}

        smpld = self.metadata.sample(1, weights=self.metadata['p'])

        def package(type_key, _batch_idx, _start_idx):
            datum = {}

            path = smpld[f"dac_path_{type_key}"].tolist()[0]
            try:
                artifact = DACFile.load(path)
            except:
                print(f"Error loading {path} or {output_file}")
                if attempt > 50:
                    raise Exception(f"Error loading {file} after {attempt} attempts.")
                return self.__getitem__(idx + random.randint(1, len(self)*10), attempt=attempt+1)

            codes = artifact.codes
            print(codes.shape)

            nb, nc, nt = codes.shape
            if _batch_idx is None:
                # grab a random batch of codes
                _batch_idx = torch.randint(0, nb, (1,)).item()
            
            codes = codes[_batch_idx, :, :]

            if _start_idx is None:
                # get a seq_len chunk out of it
                if nt <= self.seq_len:
                    _start_idx = 0
                else:
                    _start_idx = torch.randint(0, nt - self.seq_len, (1,)).item()
            else:
                assert _start_idx + self.seq_len <= nt, f"start_idx {_start_idx} + seq_len {self.seq_len} must be less than nt {nt}"

            codes = codes[:, _start_idx:_start_idx + self.seq_len]
                    
            # grab the labels
            if self.label_key is not None:
                label = smpld[self.label_key].tolist()[0]
                label = torch.tensor(self.classlist.index(label), dtype=torch.long)
            else:
                label = None
            
            codes, pad_mask = pad_if_needed(codes, self.seq_len)
            return {
                "codes": codes,
                "label": label,
                "path": path,
                "pad_mask": pad_mask,
                "batch_idx": _batch_idx,
                "start_idx": _start_idx,
            }

        batch_idx = None
        start_idx = None
        for type_key in self.type_keys:
            data[type_key] = package(type_key, batch_idx, start_idx)
            batch_idx = data[type_key]["batch_idx"]
            start_idx = data[type_key]["start_idx"]

        if len(self.type_keys) == 1:
            data = data[self.type_keys[0]]
            
        return data



def test():

    test_df = pd.DataFrame({
        "id": [0, 0, 1, 1, 2, 2, 3, 3],
        "type": ["input", "output", "input", "output", "input", "output", "input", "output"],
        "label": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "dac_path": ["dac1.dac", "dac2.dac", "dac3.dac", "dac4.dac", "dac5.dac", "dac6.dac", "dac7.dac", "dac8.dac"]
    })
    

    for (idx, row) in test_df.iterrows():
        codes = torch.randn(1, 1, 1024)
        dac = DACFile(
            codes=codes,
            chunk_length=1024,
            original_length=1024,
            input_db=torch.tensor(0),
            channels=1,
            sample_rate=44100,
            padding=False,
            dac_version="0.0.1"
        )
        dac.save(row['dac_path'])

    # save to csv
    test_df.to_csv("test.csv", index=False)


    dataset = DACDataset(
        metadata_csvs=["test.csv"],
        seq_len=113,
        split=None,
        classlist=["a", "b", "c", "d"],
        paired=True,
        main_key="input",
        type_key="type",
        id_key="id",
        label_key="label",
        class_weights=None,
        length=1000000000)
    dac = load_dac(load_path="./models/dac/weights.pth")

    # Load a sample
    for i in range(10):
        data = dataset[i]
        # dac.to('cuda')

        assert data['input']['codes'].shape == (1, 113)
        assert data['output']['codes'].shape == (1, 113)



        out.write(f"test{i}.wav")

if __name__ == "__main__":
    test()