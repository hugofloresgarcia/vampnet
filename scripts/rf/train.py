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


def spec_encode(sig: sn.Signal, window_length=255):
    sig = sn.to_mono(sig)
    sig = sn.normalize(sig, -24.0)
    spec = sn.stft(sig, hop_length=window_length // 2, window_length=window_length)

    # pick a random chunk
    chunk_idx = np.random.randint(spec.shape[0])
    spec = spec[chunk_idx:chunk_idx+1, ...]
    mag, phase = torch.abs(spec), torch.angle(spec)
    spec = torch.cat([mag, phase], dim=1)
    return spec[0]

def spec_decode(spec, window_length=255): # (shape channels, freq, time)
    mag, phase = spec.chunk(2, dim=0)
    spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
    wav = torch.istft(
        spec, hop_length=window_length // 2, n_fft=window_length, 
        window=torch.hann_window(window_length).to(spec.device)
    ).unsqueeze(0)
    return sn.Signal(wav, sr=22050)


QUERY = """
    SELECT af.path, chunk.offset, chunk.duration, af.duration as total_duration, dataset.name 
    FROM chunk 
    JOIN audio_file as af ON chunk.audio_file_id = af.id 
    JOIN dataset ON af.dataset_id = dataset.id
    WHERE dataset.name IN ('clack')
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
            df=self.df, sample_rate=22050, n_samples=self.image_size  * 2 * (self.window_length // 2), num_channels=1
        )

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(self.image_size),
            # T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        window_length = self.window_length
        sig = self.dataset[index]["sig"]
        return spec_encode(sig, window_length)



model = Unet(dim = 64, channels=2)
rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset()

plt.imshow(img_dataset[3][0], origin='lower', aspect='auto')
plt.savefig('test.png')
plt.imshow(img_dataset[55][0], origin='lower', aspect='auto')
plt.savefig('test2.png')

sig = spec_decode(img_dataset[3], img_dataset.window_length)
sn.write(sig, 'test.wav')
sig = spec_decode(img_dataset[55], img_dataset.window_length)
sn.write(sig, 'test2.wav')

trainer = Trainer(
    rectified_flow,
    batch_size=16,
    save_results_every=100,
    dataset = img_dataset,
    num_train_steps = 70_000,
    checkpoint_every=2000,
    accelerate_kwargs=dict(log_with="tensorboard", project_dir="./checkpoints/v8-clack"),
    checkpoints_folder="./checkpoints/v8-clack",
    results_folder = './results/v8-clack'   # samples will be saved periodically to this folder
)

trainer()