from typing import List
from pathlib import Path
import torch
import random
import time

import pandas as pd
from audiotools import util

from audiotools import AudioSignal
from dac.model.base import DACFile
from dac.utils import load_model as load_dac


SALAD_BOWL_WEIGHTS = {
    "Sounds of things": 1,
    "Channel, environment and background": 0.01, 
    "Natural sounds": 1.2, 
    "Human sounds": 1, 
    "Music": 1.75,
    "Animal": 1, 
    "Source-ambiguous sounds": 0.8,
}

class DACDataset(torch.utils.data.Dataset):

    def __init__(self, 
        metadata_csvs: List[str],
        seq_len: int = 1024,
        split = "train"
    ):
        
        # load the metadata csvs
        self.metadata = []
        for csv in metadata_csvs:
            self.metadata.append(pd.read_csv(csv))
        
        self.metadata = pd.concat(self.metadata)

        # filter by split
        self.metadata = self.metadata[self.metadata.split == split]

        # make a dict of family -> list of files
        self.families = self.metadata.family.unique()
        self.family_to_files = {f: [] for f in self.families}
        for _, row in self.metadata.iterrows():
            self.family_to_files[row.family].append(row.dac_path)

        # print stats
        print(f"Found {len(self.metadata)} files in {metadata_csvs}.")
        for f, files in self.family_to_files.items():
            print(f"{f}: {len(files)}")

        self.seq_len = seq_len

    def __len__(self):
        return sum([len(files) for files in self.family_to_files.values()])
    
    def __getitem__(self, idx, attempt=0):
        util.seed(idx)
        # grab a random family
        family = random.choices(self.families, weights=[SALAD_BOWL_WEIGHTS[f] for f in self.families], k=1)[0]

        # grab a file from that family
        file = random.choice(self.family_to_files[family])

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


        return {
            "codes": codes,
            "file": file,
            "pad_mask": pad_mask,
        }

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