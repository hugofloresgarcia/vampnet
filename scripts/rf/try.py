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
from vampnet.dsp.signal import spec_decode

from scripts.rf.train import ImageDataset, rectified_flow

# rectified_flow = RectifiedFlow(model)

img_dataset = ImageDataset()

trainer = Trainer(rectified_flow, dataset=img_dataset)

ckpt_path = "/home/hugo/vampnet/checkpoints/alligator_v9/checkpoint.14000.pt"
trainer.load(ckpt_path)

data_shape =  list(img_dataset[0].shape) 
data_shape[-1] = 4096

noise = torch.randn(data_shape).to("cuda") 
# give a slight shift to the left half of the noise
noise[..., :noise.shape[-1]//2] += 0.1

sampled = trainer.model.sample(noise=noise.unsqueeze(0), data_shape=data_shape)

sig = spec_decode(sampled[0])
sn.write(sig, "sample.wav")
# breakpoint()