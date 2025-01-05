import random
import vampnet
import vampnet.signal as sn
from vampnet.interface import Interface
import audiotools as at
import torch

from scripts.exp.train import VampNetTrainer

ckpt = "/home/hugo/soup/runs/debug/lightning_logs/version_180/checkpoints/last.ckpt"
codec_ckpt = "/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"

bundle = VampNetTrainer.load_from_checkpoint(ckpt, codec_ckpt=codec_ckpt) 
bundle.codec = vampnet.dac.DAC.load(codec_ckpt)
codec = bundle.codec
vn = bundle.model
vn.eval()
codec.eval()

eiface = Interface(
    codec=codec,
    vn=vn,
)

at.util.seed(0)

# load the dataset for playing w/
# from scripts.exp.train import build_datasets

# train_data, val_data = build_datasets(
#     sample_rate=codec.sample_rate, 
#     db_path="scratch/data-fast/sm.db", 
#     query="SELECT * FROM audio_file JOIN dataset where dataset.name = 'vctk'"
# )


# load an audio file
sig = sn.read_from_file("assets/example.wav")
# sig = val_data[0]["sig"]

# sig.wav = torch.cat([sig.wav, sig.wav, sig.wav], dim=-1)

# cut sig to hop length
sig = sn.trim_to_s(sig, 5.0)
sig.wav = sn.cut_to_hop_length(sig.wav, eiface.codec.hop_length)
sig = sn.normalize(sig, -16)

# move to gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
sig = sig.to(device)
codec.to(device)

codes = eiface.encode(sig.wav)
print(codes.shape)

# make a mask
mask = eiface.build_mask(codes, periodic_prompt=0, upper_codebook_mask=4)

# vamp on the codes
# chop off, leave only the top  codebooks
z = codes
z = z[:, : vn.n_codebooks, :]
mask = mask[:, : vn.n_codebooks, :]

# apply the mask
from vampnet.mask import apply_mask
z = apply_mask(z, mask, vn.mask_token)
with torch.autocast(device,  dtype=torch.bfloat16):
    zv = vn.generate(
        codes=z,
        temperature=1.0,
        # mask_temperature=1.0,
        typical_filtering=False,
        typical_mass=0.15,
        typical_min_tokens=64,
        sampling_steps=12, 
        causal_weight=0.0
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