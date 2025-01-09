import random
from vampnet.interface import Interface
import torch

import vampnet
import vampnet.signal as sn
from vampnet.util import first_dict_value, seed
from vampnet.mask import apply_mask

from vampnet.train import VampNetTrainer

# pick a device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"
seed(0)

# load a pretrained model bundle
bundle = VampNetTrainer.from_pretrained("hugggof/vampnetv2-mode-vampnet_rms-latest") 

codec = bundle.codec # grab the codec
vn = bundle.model # and the vampnet
controller = bundle.controller # and the controller

# create an interface with our pretrained shizzle
eiface = Interface(
    codec=codec,
    vn=vn,
    controller=controller
)
eiface.to(device)

# load an audio file
sig = sn.read_from_file("assets/example.wav")

# preprocess the signal
sig = sn.trim_to_s(sig, 5.0)
sig = eiface.preprocess(sig)

# extract controls and build a mask for them
ctrls = controller.extract(sig)
ctrl_masks = eiface.build_ctrl_masks(ctrls,
    periodic_prompt=5
)

# encode the signal
codes = eiface.encode(sig.wav)
print(f"encoded to codes of shape {codes.shape}")

# make a mask for the codes
mask = eiface.build_codes_mask(codes, 
    periodic_prompt=0, upper_codebook_mask=4
)

# apply the mask
codes = apply_mask(codes, mask, vn.mask_token)

# generate!
# with torch.autocast(device,  dtype=torch.bfloat16):
gcodes = vn.generate(
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
generated_wav = eiface.decode(gcodes)
sn.write(
    sn.Signal(generated_wav, sig.sr),
    "scratch/generated.wav"
)