from pathlib import Path
import time
import os
from contextlib import contextmanager
import random

import numpy as np
import audiotools as at
from audiotools import AudioSignal
import argbind
import shutil
import torch
import yaml


from vampnet.interface import Interface, signal_concat
from vampnet import mask as pmask

from ttutil import log

# TODO: incorporate discord bot (if mem allows)
# in a separate thread, send audio samples for listening
# and send back the results
# as well as the params for sampling
# also a command that lets you clear the current signal 
# if you want to start over


device = "cuda" if torch.cuda.is_available() else "cpu"

VAMPNET_DIR = Path(".").resolve()

@contextmanager
def chdir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

def load_interface(model_choice="default") -> Interface:
    with chdir(VAMPNET_DIR):


        # populate the model choices with any interface.yml files in the generated confs
        MODEL_CHOICES = {
            "default": {
                "Interface.coarse_ckpt": "models/vampnet/coarse.pth", 
                "Interface.coarse2fine_ckpt": "models/vampnet/c2f.pth",
                "Interface.codec_ckpt": "models/vampnet/codec.pth",
            }
        }
        generated_confs = Path("conf/generated")
        for conf_file in generated_confs.glob("*/interface.yml"):
            with open(conf_file) as f:
                _conf = yaml.safe_load(f)

                # check if the coarse, c2f, and codec ckpts exist
                # otherwise, dont' add this model choice
                if not (
                    Path(_conf["Interface.coarse_ckpt"]).exists() and 
                    Path(_conf["Interface.coarse2fine_ckpt"]).exists() and 
                    Path(_conf["Interface.codec_ckpt"]).exists()
                ):
                    continue

                MODEL_CHOICES[conf_file.parent.name] = _conf

    interface = Interface(
        device=device, 
        coarse_ckpt=MODEL_CHOICES[model_choice]["Interface.coarse_ckpt"],
        coarse2fine_ckpt=MODEL_CHOICES[model_choice]["Interface.coarse2fine_ckpt"],
        codec_ckpt=MODEL_CHOICES[model_choice]["Interface.codec_ckpt"],
    )

    interface.model_choices = MODEL_CHOICES
    interface.to("cuda" if torch.cuda.is_available() else "cpu")
    return interface

def load_model(interface: Interface, model_choice: str):
    interface.reload(
        interface.model_choices[model_choice]["Interface.coarse_ckpt"],
        interface.model_choices[model_choice]["Interface.coarse2fine_ckpt"],
    )

def ez_variation(
        interface,
        sig: AudioSignal,
        seed: int = None, 
        model_choice: str = None,  
    ):
    t0 = time.time()
    
    if seed is None:
        seed = int(torch.randint(0, 2**32, (1,)).item())
    at.util.seed(seed)

    # reload the model if necessary
    if model_choice is not None:
        load_model(interface, model_choice)

    # SAMPLING MASK PARAMS, hard code for now, we'll prob want a more preset-ey thing for the actual thin
    # we probably honestly just want to oscillate between the same 4 presets
    # in a predictable order such that they have a predictable outcome
    periodic_p = random.choice([3])
    n_mask_codebooks = 3
    sampletemp = random.choice([1.0,])
    dropout = random.choice([0.0, 0.0])

    top_p = None # NOTE: top p may be the culprit behind the collapse into single pitches. 

    # parameters for the build_mask function
    build_mask_kwargs = dict(
        rand_mask_intensity=1.0,
        prefix_s=0.0,
        suffix_s=0.0,
        periodic_prompt=int(periodic_p),
        periodic_prompt2=int(periodic_p),
        periodic_prompt_width=1,
        _dropout=dropout,
        upper_codebook_mask=int(n_mask_codebooks), 
        upper_codebook_mask_2=int(n_mask_codebooks),
    )

    # parameters for the vamp function
    vamp_kwargs = dict(
        temperature=sampletemp,
        typical_filtering=True, 
        typical_mass=0.15, 
        typical_min_tokens=64, 
        top_p=top_p,
        seed=seed,
        sample_cutoff=1.0,
    )

    # save the mask as a txt file
    interface.set_chunk_size(10.0)
    sig, mask, codes = interface.vamp(
        sig, 
        batch_size=1,
        feedback_steps=1,
        time_stretch_factor=1,
        build_mask_kwargs=build_mask_kwargs,
        vamp_kwargs=vamp_kwargs,
        return_mask=True,
    )

    log(f"vamp took {time.time() - t0} seconds")
    return sig



def main():
    import tqdm

    interface = load_interface()
    sig = AudioSignal.excerpt("assets/example.wav", duration=7.0)
    sig = interface.preprocess(sig)
    sig.write('ttout/in.wav')
    insig = sig.clone()

    fdbk_every = 4
    fdbk = 0.5

    for i in tqdm.tqdm(range(1000)): 
        sig = ez_variation(interface, sig, model_choice="orchestral")
        sig.write(f'ttout/out{i}.wav')
    

if __name__ == "__main__":
    main()