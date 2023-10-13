from typing import List
from pathlib import Path
import torch
import random
import time

from audiotools import AudioSignal
from dac.model.base import DACFile
from dac.utils import load_model as load_dac

class DACDataset(torch.utils.data.Dataset):

    def __init__(self, 
        paths: List[str], 
        seq_len: int = 1024, 
        shuffle: bool = True
    ):
        self.paths = [Path(p) for p in paths]
        self.seq_len = seq_len

        self.files = list()
        for path in self.paths:
            self.files.extend(list(path.glob("**/*.dac")))

        if shuffle:
            random.shuffle(self.files)
        
        print(f"Found {len(self.files)} files in {paths}.")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # grab a random file
        file = self.files[idx % len(self.files)]

        artifact = DACFile.load(file)

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