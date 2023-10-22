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


class DACDataset(torch.utils.data.Dataset):

    def __init__(self, 
        metadata_csvs: List[str] = None,
        seq_len: int = 1024,
        split: str = None, 
        classlist: list = None,
        class_key: str = "label",
        class_weights: dict = None, 
    ):
        assert metadata_csvs is not None, "Must provide metadata_csvs"
        assert split is not None, f"split must be provided but got {split}"
        
        # load the metadata csvs
        self.metadata = []
        for csv in metadata_csvs:
            self.metadata.append(pd.read_csv(csv))
        
        self.metadata = pd.concat(self.metadata)
        print(f"loaded metadata with {len(self.metadata)} rows")

        # filter by split
        if split != "all":
            self.metadata = self.metadata[self.metadata['split'] == split]
        print(f"resolved split: {split}")

        self.class_key = class_key
        assert class_key in self.metadata.columns, f"Class key {class_key} not in metadata columns {self.metadata.columns}"

        # resolve classlist
        if classlist is not None:
            self.classlist = classlist
            for _cls in classlist:
                tru_classlist = self.metadata[self.class_key].unique().tolist()
                assert _cls in tru_classlist, f"Class {_cls} not in metadata classlist {tru_classlist}"

            # filter the metadata by the classlist
            self.metadata = self.metadata[self.metadata[self.class_key].isin(classlist)]
        else:
            self.classlist = self.metadata[class_key].unique().tolist()
        print(f"resolved classlist: {self.classlist}")
        print(f'metadata now has {len(self.metadata)} rows')


        # load the class weights for sampling if any
        # resolve class weights
        if class_weights is not None:
            assert class_weights.keys() == self.classlist, f"Class weights keys {class_weights.keys()} do not match classlist {self.classlist}"
            self.class_weights = class_weights
        else:
            self.class_weights = None
        print(f"resolved class weights: {self.class_weights}")
    
        self.seq_len = seq_len

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx, attempt=0):
        util.seed(idx)
        data = {}

        # 1. Sample a class according to the weights if class_cond_weights is provided
        if self.class_weights is not None:
            selected_class = random.choices(self.classlist, weights=list(self.class_weights.values()), k=1)[0]
            
            # 2. Pick a file from that class, add the classname to the data{}
            class_files = self.metadata[self.metadata[self.class_key] == selected_class]
            if class_files.empty:
                raise ValueError(f"No files found for class: {selected_class}")
            
            file = random.choice(class_files['dac_path'].tolist())  
        else:
            smpld = self.metadata.sample(1)
            file = smpld['dac_path'].tolist()[0] 

            selected_class = smpld[self.class_key].tolist()[0]
            data['label'] = torch.tensor(self.classlist.index(selected_class), dtype=torch.long)

        try:
            artifact = DACFile.load(file)
        except:
            print(f"Error loading {file}")
            if attempt > 50:
                raise Exception(f"Error loading {file} after {attempt} attempts.")
            return self.__getitem__(idx + random.randint(1, len(self)*10), attempt=attempt+1)

        # shape (channels, num_chunks, seq_len)
        codes = artifact.codes

        # grab a random batch of codes
        nb, nc, nt = codes.shape
        batch_idx = torch.randint(0, nb, (1,)).item()
        codes = codes[batch_idx, :, :].unsqueeze(0)

        # get a seq_len chunk out of it
        if nt <= self.seq_len:
            start_idx = 0
        else:
            start_idx = torch.randint(0, nt - self.seq_len, (1,)).item()
        codes = codes[:, :, start_idx:start_idx + self.seq_len]

        # if our seq is too short, then we need to pad it
        pad_mask = torch.ones(1, self.seq_len).int()
        if codes.shape[-1] < self.seq_len:
            pad_len = self.seq_len - codes.shape[-1]
            codes = torch.nn.functional.pad(codes, (0, pad_len))
            pad_mask[:, -pad_len:] = 0

        data.update({
            "codes": codes,
            "file": file,
            "pad_mask": pad_mask,
        })

        return data

    @staticmethod
    def collate(batch):
        codes = torch.cat([b["codes"] for b in batch], dim=0)
        pad_mask = torch.cat([b["pad_mask"] for b in batch], dim=0)
        file = [b["file"] for b in batch]
        return {
            "codes": codes,
            "ctx_mask": pad_mask,
            "file": file,
        }



def test():
    dataset = DACDataset([Path("./data/codes-mono/prosound")])
    dac = load_dac(load_path="./models/dac/weights.pth")

    # Load a sample
    for i in range(10):
        data = dataset[i]
        dac.to('cuda')

        # Decode the sample
        with torch.inference_mode():
            z, _, _ = dac.quantizer.from_codes(data["codes"].to('cuda'))
            out = AudioSignal(
                dac.decode(z).detach().cpu(), 
                sample_rate=dac.sample_rate
            )


        out.write(f"test{i}.wav")

if __name__ == "__main__":
    test()