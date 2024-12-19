import random
import vampnet
import vampnet.signal as sn
from vampnet.interface import EmbeddedInterface
import audiotools as at
import torch

from scripts.exp.train import VampNetTrainer

ckpt = "/home/hugo/soup/runs/debug/lightning_logs/version_23/checkpoints/epoch=16-step=119232.ckpt"
codec_ckpt = "/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"

bundle = VampNetTrainer.load_from_checkpoint(ckpt, codec_ckpt=codec_ckpt) 
codec = bundle.codec
vn = bundle.model
vn.eval()
codec.eval()
eiface = EmbeddedInterface(
    codec=codec,
    coarse=vn,
)

# load the dataset for playing w/
from scripts.exp.train import build_datasets

train_data, val_data = build_datasets(
    sample_rate=codec.sample_rate, 
    db_path="scratch/data-fast/sm.db", 
    query="SELECT * FROM audio_file JOIN dataset where dataset.name = 'vctk'"
)


# load an audio file
# sig = sn.read_from_file("assets/example.wav")
sig = val_data[0]["sig"]

# cut sig to hop length
sig.wav = sn.cut_to_hop_length(sig.wav, eiface.codec.hop_length)

# move to gpu
sig = sig.to("cuda")
codec.to("cuda")

codes = eiface.encode(sig.wav)
print(codes.shape)

# make a mask
mask = eiface.build_mask(codes)

# vamp on the codes
# chop off, leave only the top  codebooks
z = codes
z = z[:, : vn.n_codebooks, :]
mask = mask[:, : vn.n_codebooks, :]

# apply the mask
from vampnet.mask import apply_mask
z = apply_mask(z, mask, vn.mask_token)
with torch.autocast("cuda",  dtype=torch.bfloat16):
    zv = vn.generate(
        codes=z,
        temperature=1.0,
        typical_filtering=False,
        typical_mass=0.15,
        typical_min_tokens=64,
        seed=0,
        sampling_steps=24
    )


# decode
wav = eiface.decode(codes)
outsig = sn.Signal(wav, sig.sr)
print(wav.shape)

# write to file
sn.write(outsig, "scratch/ex-recons.wav")

# save generated
generated_wav = eiface.decode(zv)
sn.write(sn.Signal(generated_wav, sig.sr), "scratch/ex-gen.wav")