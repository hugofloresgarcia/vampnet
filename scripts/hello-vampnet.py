import random
from vampnet.interface import Interface
import torch

import vampnet
import vampnet.signal as sn
from vampnet.util import first_dict_value, seed
from vampnet.mask import apply_mask

from scripts.train import VampNetTrainer

# pick a device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
seed(0)

# load a pretrained model bundle
bundle = VampNetTrainer.from_pretrained("hugggof/vampnetv2-mode-vampnet_rms-latest") 

codec = bundle.codec # grab the codec
vn = bundle.model # and the vampnet
controller = bundle.controller # and the controller

# eval mode!
vn.eval()
codec.eval()

# create an interface with our pretrained shizzle
eiface = Interface(
    codec=codec,
    vn=vn,
    controller=controller
)

# load an audio file
sig = sn.read_from_file("assets/example.wav")

# preprocess the signal
sig = sn.trim_to_s(sig, 5.0)
sig.wav = sn.cut_to_hop_length(sig.wav, eiface.codec.hop_length)
sig = sn.normalize(sig, -16) # TODO: we should refactor this magic number
sig = sig.to(device)

# extract controls and build a mask for them
ctrls = controller.extract(sig)
ctrl_mask =  eiface.build_mask(
    first_dict_value(ctrls), 
    periodic_prompt=7, 
    upper_codebook_mask=1)[:, 0, :]
ctrl_masks = {
    k: ctrl_mask for k in ctrls.keys()
}

# move to gpu
codec.to(device)

# encode the signal
codes = eiface.encode(sig.wav)
print(f"encoded to codes of shape {codes.shape}")

# make a mask for the codes
mask = eiface.build_mask(codes, periodic_prompt=0, upper_codebook_mask=4)

# apply the mask
codes = apply_mask(codes, mask, vn.mask_token)

# generate!
with torch.autocast(device,  dtype=torch.bfloat16):
    zv = vn.generate(
        codes=codes,
        temperature=1.0,
        mask_temperature=100.0,
        typical_filtering=True,
        typical_mass=0.15,
        ctrls=ctrls,
        ctrl_masks=ctrl_masks,
        typical_min_tokens=64,
        sampling_steps=[16, 8, 4, 4],
        # sampling_steps=16,
        causal_weight=0.0,
        debug=False
    )

# decode
wav = eiface.decode(codes)
outsig = sn.Signal(wav, sig.sr)
print(wav.shape)

# write the reconstructed signal
sn.write(outsig, "scratch/reconstructed.wav")

# write the generated signal
generated_wav = eiface.decode(zv)
sn.write(
    sn.Signal(generated_wav, sig.sr),
    "scratch/generated.wav"
)