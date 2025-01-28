from pathlib import Path

import torch
from torch import nn
import numpy as np

from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import soundmaterial as sm
from soundmaterial.dataset import Dataset
import vampnet.dsp.signal as sn
import pandas as pd
from einops import rearrange


DATASET = "alligator"

QUERY = f"""
    SELECT af.path, chunk.offset, chunk.duration, af.duration as total_duration, dataset.name 
    FROM chunk 
    JOIN audio_file as af ON chunk.audio_file_id = af.id 
    JOIN dataset ON af.dataset_id = dataset.id
    WHERE dataset.name IN ('{DATASET}')
"""



class ImageDataset:

    def __init__(self, 
            window_length=255, 
            augment_horizontal_flip=False, 
        ):
        self.window_length = window_length
        self.image_size = window_length // 2 + 1
        # the path to our database
        db_path = "./sm.db"

        # connect to our database
        conn = sm.connect(db_path)

        # find all the wav files
        query = QUERY

        # Create a subset of the database
        self.df = pd.read_sql_query(query, conn)
        self.df = self.df.sample(n=len(self.df))

        self.dataset = Dataset(
            df=self.df, sample_rate=22050, n_samples=self.image_size  * 2 * (self.window_length // 2), num_channels=1,
            use_chunk_table=True
        )

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(self.image_size),
            # T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        window_length = self.window_length
        sig = self.dataset[index]["sig"]
        spec =  sn.spec_encode(sig, window_length)
        return spec



model = Unet(dim = 64, channels=2)
rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset()

plt.imshow(img_dataset[3][0], origin='lower', aspect='auto')
plt.savefig('test.png')
plt.imshow(img_dataset[55][0], origin='lower', aspect='auto')
plt.savefig('test2.png')

sig = sn.spec_decode(img_dataset[3], img_dataset.window_length)
sn.write(sig, 'test.wav')
sig = sn.spec_decode(img_dataset[55], img_dataset.window_length)
sn.write(sig, 'test2.wav')

# save the 100 first audio files
from pathlib import Path
Path("audio_files").mkdir(exist_ok=True)
for i in range(100):
    spec = img_dataset[i]
    sig = sn.spec_decode(spec, img_dataset.window_length)
    sn.write(sig, f"audio_files/{i}.wav")

version = f"{DATASET}_v9"
save_dir = Path(f"checkpoints/{version}")
trainer = Trainer(
    rectified_flow,
    batch_size=16,
    save_results_every=1000,
    dataset = img_dataset,
    num_train_steps = 70_000,
    checkpoint_every=2000,
    accelerate_kwargs=dict(log_with="tensorboard", project_dir=str(save_dir)),
    checkpoints_folder=str(save_dir),
    results_folder =str(save_dir/"results")   # samples will be saved periodically to this folder
)

if __name__ == "__main__":
    trainer()