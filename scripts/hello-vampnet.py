import random
from vampnet.interface import Interface
import torch

import vampnet
import vampnet.signal as sn
from vampnet.util import first_dict_value, seed
from vampnet.mask import apply_mask

from scripts.train import VampNetTrainer

seed(0)

bundle = VampNetTrainer.from_pretrained("hugggof/vampnetv2-mode-vampnet_rms-latest") 
codec = bundle.codec
vn = bundle.model
vn.eval()
codec.eval()

eiface = Interface(
    codec=codec,
    vn=vn,
    controller=bundle.controller
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load an audio file
sig = sn.read_from_file("assets/example.wav")

# cut sig to hop length
sig = sn.trim_to_s(sig, 5.0)
sig.wav = sn.cut_to_hop_length(sig.wav, eiface.codec.hop_length)
sig = sn.normalize(sig, -16)
sig = sig.to(device)

# extract controls
ctrls = eiface.controller.extract(sig)
ctrl_mask =  eiface.build_mask(
    first_dict_value(ctrls), 
    periodic_prompt=100, 
    upper_codebook_mask=1)[:, 0, :]
ctrl_masks = {
    k: ctrl_mask for k in ctrls.keys()
}

# move to gpu
codec.to(device)

codes = eiface.encode(sig.wav)
print(codes.shape)

# make a mask
mask = eiface.build_mask(codes, periodic_prompt=0, upper_codebook_mask=4)

codes = apply_mask(codes, mask, vn.mask_token)


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
        debug=True
    )


# decode
wav = eiface.decode(codes)
outsig = sn.Signal(wav, sig.sr)
print(wav.shape)

# write to file
sn.write(outsig, "scratch/reconstructed.wav")

# save generated
generated_wav = eiface.decode(zv)
sn.write(sn.Signal(generated_wav, sig.sr), f"scratch/generated.wav")