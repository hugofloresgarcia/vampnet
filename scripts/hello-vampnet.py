import random
from vampnet.interface import Interface
import torch

import vampnet
import vampnet.dsp.signal as sn
from vampnet.util import first_dict_value, seed
from vampnet.mask import apply_mask

from vampnet.train import VampNetTrainer

# pick a device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" 
# seed(0)

# load a pretrained model bundle
bundle = VampNetTrainer.from_pretrained("hugggof/vampnetv2-tria-d774-l8-h8-mode-vampnet_rms-hchroma-36c-top3-latest") 

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
sig = sn.read_from_file("assets/voice-prompt.wav")
sig = sn.trim_to_s(sig, 5.0)
ldns = sn.loudness(sig)
sig = eiface.preprocess(sig)

# load a drum sample
sig_spl = sn.read_from_file("assets/noodle.wav", duration=1.0)
sig_spl = eiface.preprocess(sig_spl)

# extract onsets, for our onset mask
onset_idxs = sn.onsets(sig, hop_length=codec.hop_length)

# extract controls and build a mask for them
ctrls = controller.extract(sig)
ctrl_masks = {}
ctrl_masks["rms"] = eiface.rms_mask(
    ctrls["rms"], onset_idxs=onset_idxs, 
    periodic_prompt=7, drop_amt=0.3
)
ctrl_masks["hchroma-36c-top3"] = torch.zeros_like(ctrl_masks["rms"])
ctrl_masks["hchroma-36c-top3"] = ctrl_masks["rms"]

# encode the signal
codes = eiface.encode(sig.wav)
print(f"encoded to codes of shape {codes.shape}")

# make a mask for the codes
mask = eiface.build_codes_mask(codes, 
    periodic_prompt=0, upper_codebook_mask=0
)

# encode the sample
codes_spl = eiface.encode(sig_spl.wav)
print(f"encoded to codes of shape {codes_spl.shape}")

# add sample to bundle
codes, mask, ctrls, ctrl_masks = eiface.add_sample(
    spl_codes=codes_spl, codes=codes, 
    cmask=mask, ctrls=ctrls, ctrl_masks=ctrl_masks
)

# apply the mask
mcodes = apply_mask(codes, mask, vn.mask_token)

# visualize the bundle
eiface.visualize(
    sig=sn.concat([sig_spl,sig]), 
    codes=mcodes, mask=mask, 
    ctrls=ctrls, ctrl_masks=ctrl_masks
)

# generate!
# with torch.autocast(device,  dtype=torch.bfloat16):
gcodes = vn.generate(
    codes=mcodes,
    temperature=1.0,
    cfg_scale=5.0,
    mask_temperature=10.0,
    typical_filtering=True,
    typical_mass=0.15,
    ctrls=ctrls,
    ctrl_masks=ctrl_masks,
    typical_min_tokens=128,
    sampling_steps=24 if vn.mode == "vampnet" else [16, 8, 4, 4],
    causal_weight=0.0,
    debug=False
)

# decode
wav = eiface.decode(codes)
outsig = sn.Signal(wav, sig.sr)
outsig = sn.normalize(outsig, ldns)
print(wav.shape)

# write the reconstructed signal
sn.write(outsig, "scratch/reconstructed.wav")

# write the generated signal
generated_wav = eiface.decode(gcodes)
sn.write(
    sn.Signal(generated_wav, sig.sr),
    "scratch/generated.wav"
)