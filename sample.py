import yaml
import argbind

import audiotools as at

from vampnet.interface import Interface
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

Interface = argbind.bind(Interface)

with open("conf/interface/spotdl.yml") as f:
    conf = yaml.safe_load(f)


with argbind.scope(conf):
    interface = Interface()
    interface.to("cuda")

loader = at.data.datasets.AudioLoader(sources=[
    "input.wav",
])

dataset = at.data.datasets.AudioDataset(
    loader,
    sample_rate=interface.codec.sample_rate,
    duration=interface.coarse.chunk_size_s,
    n_examples=200,
    without_replacement=True,
)

import numpy as np
def load_random_audio():
    index = np.random.randint(0, len(dataset))
    sig = dataset[index]["signal"]
    sig = interface.preprocess(sig)

    return sig


sig = load_random_audio()
z = interface.encode(sig)

sig.write('input.wav')

from vampnet import mask as pmask

# build the mask
mask = pmask.linear_random(z, 1.0)

print("coarse")
zv, mask_z = interface.coarse_vamp(
    z, 
    mask=mask,
    sampling_steps=36,
    temperature=8.0,
    return_mask=True, 
    typical_filtering=False, 
    # typical_mass=data[typical_mass], 
    # typical_min_tokens=data[typical_min_tokens], 
    gen_fn=interface.coarse.generate,
)

print("coarse2fine")
zv = interface.coarse_to_fine(zv, temperature=0.8)

sig = interface.to_signal(zv).cpu()
sig.write('output-t=8.wav')