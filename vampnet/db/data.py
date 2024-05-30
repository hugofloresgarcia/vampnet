
from pathlib import Path
from typing import Optional, List
import torch
import pandas as pd

import vampnet
from vampnet import mask as pmask


# if our seq is too short, then we need to pad it
def pad_if_needed(codes, seq_len):
    pad_mask = torch.ones(seq_len).int()
    if codes.shape[-1] < seq_len:
        pad_len = seq_len - codes.shape[-1]
        codes = torch.nn.functional.pad(codes, (0, pad_len))
        pad_mask[-pad_len:] = 0
    return codes, pad_mask

class VampNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str = vampnet.DATASET, 
        codes_key: str = vampnet.CODES_KEY,
        ctrl_keys: List[str] = vampnet.CTRL_KEYS, 
        seq_len: int = vampnet.SEQ_LEN, 
        split: Optional[str] = None, 
        max_len: Optional[int] = None
    ):
        conn = vampnet.db.conn(read_only=True)
        # get the dataset id
        # breakpoint()
        print(f"Loading dataset {dataset}")
        dataset_id, root = conn.execute(f"""
            SELECT id, root 
            FROM dataset 
            WHERE name = '{dataset}'
        """).fetchone()
        self.root = Path(root)

        # TODO : is there a way to do this in one query?
        if split is None:
            df = conn.execute(
                f"""
                SELECT cs.id, cs.path, cs.audio_file_id, cs.name, cs.hop_size, cs.num_frames
                FROM ctrl_sig as cs
                JOIN audio_file as af ON af.id = cs.audio_file_id
                WHERE af.dataset_id = {dataset_id}
                """, 
            ).df()
        else:
            df = conn.execute(
                f"""
                SELECT cs.id, cs.path, cs.audio_file_id, cs.name, cs.hop_size, cs.num_frames, s.split 
                FROM ctrl_sig as cs
                JOIN audio_file as af ON af.id = cs.audio_file_id
                JOIN split as s ON s.audio_file_id = af.id
                WHERE af.dataset_id = {dataset_id} and s.split = '{split}'
                """, 
            ).df()
            print(f"Loaded {len(df)} rows from the database for split {split}")
        self.split = split

        #  shuffle, then take the first max_len
        if max_len is not None:
            df = df.sample(frac=1).iloc[:max_len]
        else:
            df = df.sample(frac=1)
        print(f"Using {len(df)} rows for split {split}")

        # make a df for the codes key and for each ctrl key
        # and we wanna use audio_file_id as the index for all
        # of these dataframes
        self.dfs = {}
        for key in [codes_key] + ctrl_keys:
            self.dfs[key] = df[df["name"] == key].set_index("audio_file_id")

        # make pd dataframe with just the audio_file_id column that we'll scramble and sample from
        self.index_df = pd.DataFrame({"audio_file_id": df["audio_file_id"].unique()})
        self.index_df_samples = self.index_df.copy().sample(frac=1)
        
        self.seq_len = seq_len
        self.ctrl_keys = ctrl_keys
        self.dataset_name = dataset

        from vampnet.controls import load_control_signal_extractors
        self.Controls = {
            c.name: c for c in load_control_signal_extractors()
        }

    def __len__(self):
        # we can find roughly the length by counting all of the frames for our codes key
        # and dividing by the seq len
        # HACK: if val, then just count ids, not frames
        if self.split == "val":
            return len(self.index_df_samples)
        else:
            n_frames = self.dfs[vampnet.CODES_KEY]["num_frames"].sum()
            return n_frames // self.seq_len
    
    def __getitem__(self, idx):
        # take a sample out of the index df, remove it from the df
        # if it's empty, reset the index df and reshuffle
        audio_file_id = self.index_df_samples.iloc[0]["audio_file_id"]
        self.index_df_samples = self.index_df_samples.iloc[1:]
        if len(self.index_df_samples) == 0:
            self.index_df_samples = self.index_df.copy().sample(frac=1)
        
        # load our codes key, sample an offset and return the slice
        code_data = self.dfs[vampnet.CODES_KEY].loc[audio_file_id]
        # get the total number of frames
        n_frames = code_data["num_frames"]
        # sample a random offset
        if n_frames > self.seq_len:
            offset = torch.randint(0, n_frames - self.seq_len, (1,)).item()
        else:
            offset = 0
        
        # load from the path
        from vampnet.controls.codec import DACControl
        codes = DACControl.load(vampnet.CACHE_PATH / self.dataset_name / code_data["path"], offset=offset, num_frames=self.seq_len)

        # load the control signals
        ctrls = {}
        for key in self.ctrl_keys:
            ctrls[key] = self.Controls[key].load(
                vampnet.CACHE_PATH / self.dataset_name / self.dfs[key].loc[audio_file_id]["path"], offset=offset, num_frames=self.seq_len
            )

        # pick a channel 
        channel = torch.randint(0, codes.ctrl.shape[0], (1,)).item()
        codes.ctrl, cmask = pad_if_needed(codes.ctrl, self.seq_len)
        codes.ctrl = codes.ctrl[channel, :, :]

        if len(ctrls) != 0:
            # concat all control tensors
            ctrl = torch.cat([
                ctrls[key].ctrl for key in self.ctrl_keys
            ], dim=1)

            # make a ctx mask
            ctrl, cmask = pad_if_needed(ctrl, self.seq_len)
            ctrl = ctrl[:, channel, :]
        else:
            ctrl = None

        return dict(
            codes=codes.ctrl.long(),
            ctrls=ctrl,
            ctx_mask=cmask.long(),
        )
    
    @property
    def control_dim(self):
        # get a random item
        item = self[0]
        # repopulate the index df
        self.index_df_samples = self.index_df.copy().sample(frac=1)
        return item["ctrls"].shape[-1]

    @staticmethod   
    def collate(batch):
        out = {}
        for key in batch[0].keys():
            val = batch[0][key]
            if isinstance(val, torch.Tensor):
                out[key] = torch.stack([item[key] for item in batch])
            elif isinstance(val, dict):
                out[key] = VampNetDataset.collate([item[key] for item in batch])
            else:
                out[key] = [item[key] for item in batch]
        return out


def build_datasets(dataset=vampnet.DATASET, split=True):
    if split:
        return VampNetDataset(dataset=dataset, split="train"), VampNetDataset(dataset=dataset, split="val", max_len=10000), VampNetDataset(dataset=dataset, split="test", max_len=10000)
    else:
        print(f"fine-tuning, will validate on the training data!!!")
        return VampNetDataset(dataset=dataset, ), VampNetDataset(dataset=dataset,), VampNetDataset(dataset=dataset, )
