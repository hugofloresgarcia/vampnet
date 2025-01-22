import random
from vampnet.interface import Interface
import torch

import vampnet
import vampnet.dsp.signal as sn
from vampnet.util import first_dict_value, seed
from vampnet.mask import apply_mask

from vampnet.train import VampNetTrainer

# pick a device and seed
ckpt = "hugggof/vampnetv2-tria-d1026-l8-h8-mode-vampnet_rms-median-latest"
device = "cuda" if torch.cuda.is_available() else "cpu"

sig_spl = sn.read_from_file("assets/noodle.wav", duration=1.0)
sig = sn.read_from_file("assets/a-beautiful-loop.wav", duration=5.0)
sig = sn.to_mono(sig)
# seed(0)

# load a pretrained model bundle
bundle = VampNetTrainer.from_pretrained(ckpt) 

codec = bundle.codec # grab the codec
vn = bundle.model # and the vampnet
controller = bundle.controller # and the controller
# controller.controllers["rms-median"].median_filter_size=20

# create an interface with our pretrained shizzle
eiface = Interface(
    codec=codec,
    vn=vn,
    controller=controller
)
eiface.to(device)

# preprocess the signal (but remember the loudness, we'll need it later)
ldns = sn.loudness(sig)
sig = eiface.preprocess(sig)

# load a drum sample
if sig_spl is not None:
    sig_spl = eiface.preprocess(sig_spl)

# extract onsets, for our onset mask
onset_idxs = sn.onsets(sig, hop_length=codec.hop_length)

# extract controls and build a mask for them
ctrls = eiface.controller.extract(sig)
ctrl_masks = {}
if len(ctrls) > 0:
    rms_key = [k for k in ctrls.keys() if "rms" in k][0]
    ctrl_masks[rms_key] = eiface.rms_mask(
        ctrls[rms_key], onset_idxs=onset_idxs, 
        periodic_prompt=0, 
        drop_amt=0.0
    )
    # use the rms mask for the other controls
    for k in ctrls.keys():
        if k != rms_key:
            ctrl_masks[k] = ctrl_masks[rms_key]
            # alternatively, zero it out
            # ctrl_masks[k] = torch.zeros_like(ctrl_masks["rms"])

# encode the signal
codes = eiface.encode(sig.wav)
print(f"encoded to codes of shape {codes.shape}")

# make a mask for the codes
mask = eiface.build_codes_mask(codes, 
    periodic_prompt=0, upper_codebook_mask=0
)

# encode the sample
if sig_spl is not None:
    codes_spl = eiface.encode(sig_spl.wav)
    print(f"encoded to codes of shape {codes_spl.shape}")

    # add sample to bundle
    codes, mask, ctrls, ctrl_masks = eiface.add_sample(
        spl_codes=codes_spl, codes=codes, 
        cmask=mask, ctrls=ctrls, ctrl_masks=ctrl_masks
    )

# apply the mask
mcodes = apply_mask(codes, mask, vn.mask_token)

vizsig = sn.concat([sig_spl, sig]) if sig_spl is not None else sig
# visualize the bundle
eiface.visualize(
    sig=vizsig, 
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