from pathlib import Path

import torch
from torch import nn
import numpy as np

from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer

import soundmaterial as sm
from soundmaterial.dataset import Dataset
import vampnet.dsp.signal as sn
import pandas as pd
from einops import rearrange
import torchvision.transforms as T


def decode(spec): # (shape channels, freq, time)
    mag, phase = spec.chunk(2, dim=0)
    # mag = torch.exp(mag)
    # mag = torch.clip(mag, max=1e2)
    # x = torch.cos(phase)
    # y = torch.sin(phase)
    # recalculating phase here does not produce anything new
    # only costs time
    # phase = torch.atan2(y, x)
    # spec = mag * torch.exp(phase * 1j)
    # better directly produce the complex value 
    # spec = mag * (x + 1j * y)
    phase = phase * np.pi
    spec = torch.expm1(mag) * (torch.cos(phase) + 1j * torch.sin(phase))
    wav = torch.istft(
        spec, hop_length=self.window_length // 2, n_fft=self.window_length, 
        window=torch.hann_window(self.window_length).to(spec.device)
    ).unsqueeze(0)
    return sn.Signal(wav, sr=22050)

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
        query = "SELECT * FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id WHERE dataset.name = 'bbc-subset'"

        # Create a subset of the database
        self.df = pd.read_sql_query(query, conn)
        self.df = self.df.sample(n=len(self.df))

        self.dataset = Dataset(
            df=self.df, sample_rate=22050, n_samples=self.image_size * (self.window_length // 2), num_channels=1
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
        sig = sn.to_mono(sig)
        sig = sn.normalize(sig, -24.0)
        spec = sn.stft(sig, hop_length=window_length // 2, window_length=window_length)
        # if the spec is too small, repeat it
        # while spec.shape[-1] < self.image_size:
        #     spec = torch.cat([spec, spec], dim=-1)
        # spec = torch.unfold_copy(
        #     spec, dimension=-1, size=self.image_size, step=window_length
        # )
        # spec = rearrange(spec, "b c f chnk t -> (b chnk) c f t")

        # pick a random chunk
        chunk_idx = np.random.randint(spec.shape[0])
        spec = spec[chunk_idx:chunk_idx+1, ...]
        mag, phase = torch.abs(spec), torch.angle(spec)
        # normalize the magnitude
        mag = torch.log1p(mag)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)

        phase = phase / np.pi

        # mag = torch.log1p(mag)
        spec = torch.cat([mag, phase], dim=1)

        # spec = self.transform(spec)
        return spec[0]

    def decode(self, spec): # (shape channels, freq, time)
        mag, phase = spec.chunk(2, dim=0)
        # mag = torch.exp(mag)
        # mag = torch.clip(mag, max=1e2)
        # x = torch.cos(phase)
        # y = torch.sin(phase)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # spec = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        # spec = mag * (x + 1j * y)
        phase = phase * np.pi
        spec = torch.expm1(mag) * (torch.cos(phase) + 1j * torch.sin(phase))
        wav = torch.istft(
            spec, hop_length=self.window_length // 2, n_fft=self.window_length, 
            window=torch.hann_window(self.window_length).to(spec.device)
        ).unsqueeze(0)
        return sn.Signal(wav, sr=22050)



model = Unet(dim = 64, channels=2)

rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset()

# save as an image (its a tensor)
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# s0 = s0.permute(1, 2, 0).detach().cpu().numpy()
# s0 = np.clip(s0, 0, 1)
# s0 = (s0 * 255).astype(np.uint8)
# img = Image.fromarray(s0)
# img.save('test.png')
plt.imshow(img_dataset[2][0], origin='lower', aspect='auto')
plt.savefig('test.png')
plt.imshow(img_dataset[11][0], origin='lower', aspect='auto')
plt.savefig('test2.png')

sig = img_dataset.decode(img_dataset[2])
sn.write(sig, 'test.wav')
sig = img_dataset.decode(img_dataset[11])
sn.write(sig, 'test2.wav')

trainer = Trainer(
    rectified_flow,
    batch_size=32,
    save_results_every=100,
    dataset = img_dataset,
    num_train_steps = 70_000,
    checkpoint_every=5000,
    accelerate_kwargs=dict(log_with="tensorboard", project_dir="./checkpoints/v7"),
    checkpoints_folder="./checkpoints/v7",
    results_folder = './results/v7'   # samples will be saved periodically to this folder
)

trainer()