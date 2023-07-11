from pathlib import Path
import random
from typing import List
import tempfile
import subprocess

import argbind
from tqdm import tqdm
import torch

from vampnet.interface import Interface
from vampnet import mask as pmask
import audiotools as at

Interface: Interface = argbind.bind(Interface)



def calculate_bitrate(
        interface, num_codebooks, 
        downsample_factor
    ):
    bit_width = 10
    sr = interface.codec.sample_rate
    hop = interface.codec.hop_size
    rate = (sr / hop) * ((bit_width * num_codebooks) / downsample_factor)
    return rate

def baseline(sig, interface):
    return interface.preprocess(sig)

def reconstructed(sig, interface):
    return interface.to_signal(
        interface.encode(sig)
    )

def coarse2fine(sig, interface):
    z = interface.encode(sig)
    z = z[:, :interface.c2f.n_conditioning_codebooks, :]

    z = interface.coarse_to_fine(z)
    return interface.to_signal(z)

class CoarseCond:

    def __init__(self, num_conditioning_codebooks, downsample_factor):
        self.num_conditioning_codebooks = num_conditioning_codebooks
        self.downsample_factor = downsample_factor

    def __call__(self, sig, interface):
        z = interface.encode(sig)
        mask = pmask.full_mask(z)
        mask = pmask.codebook_unmask(mask, self.num_conditioning_codebooks)
        mask = pmask.periodic_mask(mask, self.downsample_factor)

        zv = interface.coarse_vamp(z, mask)
        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)

def opus(sig, interface, bitrate=128):
    sig = interface.preprocess(sig)
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sig.write(f.name)

        opus_name = Path(f.name).with_suffix(".opus")
        # convert to opus
        cmd = [
            "ffmpeg", "-y", "-i", f.name, 
            "-c:a", "libopus", 
            "-b:a", f"{bitrate}", 
           opus_name
        ]
        subprocess.run(cmd, check=True)

        # convert back to wav
        output_name = Path(f"{f.name}-opus").with_suffix(".wav")
        cmd = [
            "ffmpeg", "-y", "-i", opus_name, 
            output_name
        ]

        subprocess.run(cmd, check=True)

        sig = at.AudioSignal(
            output_name, 
            sample_rate=sig.sample_rate
        )
    return sig

def mask_ratio_1_step(ratio=1.0):
    def wrapper(sig, interface):
        z = interface.encode(sig)
        mask = pmask.linear_random(z, ratio)
        zv = interface.coarse_vamp(
            z, 
            mask,
            sampling_steps=1, 
        )

        return interface.to_signal(zv)
    return wrapper

def num_sampling_steps(num_steps=1):
    def wrapper(sig, interface: Interface):
        z = interface.encode(sig)
        mask = pmask.periodic_mask(z, 16)
        zv = interface.coarse_vamp(
            z, 
            mask,
            sampling_steps=num_steps, 
        )

        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)
    return wrapper

def beat_mask(ctx_time):
    def wrapper(sig, interface):
        beat_mask = interface.make_beat_mask(
            sig,
            before_beat_s=ctx_time/2,
            after_beat_s=ctx_time/2,
            invert=True
        )

        z = interface.encode(sig)

        zv = interface.coarse_vamp(
            z, beat_mask
        )

        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)
    return wrapper

def inpaint(ctx_time):
    def wrapper(sig, interface: Interface):
        z = interface.encode(sig)
        mask = pmask.inpaint(z, interface.s2t(ctx_time), interface.s2t(ctx_time))

        zv = interface.coarse_vamp(z, mask)
        zv = interface.coarse_to_fine(zv)
        
        return interface.to_signal(zv)
    return wrapper

def token_noise(noise_amt):
    def wrapper(sig, interface: Interface):
        z = interface.encode(sig)
        mask = pmask.random(z, noise_amt)
        z = torch.where(
            mask, 
            torch.randint_like(z, 0, interface.coarse.vocab_size), 
            z
        )
        return interface.to_signal(z)
    return wrapper

EXP_REGISTRY = {}

EXP_REGISTRY["gen-compression"] = {
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    **{
        f"{n}_codebooks_downsampled_{x}x": CoarseCond(num_conditioning_codebooks=n, downsample_factor=x)
            for (n, x) in (
                (1, 1), # 1 codebook, no downsampling
                (4, 4), # 4 codebooks, downsampled 4x
                (4, 16), # 4 codebooks, downsampled 16x
                (4, 32), # 4 codebooks, downsampled 16x
            )
    }, 
    **{
        f"token_noise_{x}": mask_ratio_1_step(ratio=x)
            for x in [0.25, 0.5, 0.75]
    },

}


EXP_REGISTRY["sampling-steps"] = {
    # "codec": reconstructed,
    **{f"steps_{n}": num_sampling_steps(n)  for n in [1, 4, 12, 36, 64, 72]},
}


EXP_REGISTRY["musical-sampling"] = {
    **{f"beat_mask_{t}": beat_mask(t) for t in [0.075]}, 
    **{f"inpaint_{t}": inpaint(t) for t in [0.5, 1.0,]}, # multiply these by 2 (they go left and right)
}

@argbind.bind(without_prefix=True)
def main(
        sources=[
            "/media/CHONK/hugo/spotdl/val",
        ], 
        output_dir: str = "./samples",
        max_excerpts: int = 2000,
        exp_type: str = "gen-compression", 
        seed: int = 0,
        ext: str = [".mp3"],
    ):
    at.util.seed(seed)
    interface = Interface()

    output_dir = Path(output_dir) 
    output_dir.mkdir(exist_ok=True, parents=True)

    from audiotools.data.datasets import AudioLoader, AudioDataset

    loader = AudioLoader(sources=sources, shuffle_state=seed, ext=ext)
    dataset = AudioDataset(loader, 
        sample_rate=interface.codec.sample_rate, 
        duration=interface.coarse.chunk_size_s, 
        n_examples=max_excerpts, 
        without_replacement=True,
    )

    if exp_type in EXP_REGISTRY:
        SAMPLE_CONDS = EXP_REGISTRY[exp_type]
    else:
        raise ValueError(f"Unknown exp_type {exp_type}")


    indices = list(range(max_excerpts))
    random.shuffle(indices)
    for i in tqdm(indices):
        # if all our files are already there, skip
        done = []
        for name in SAMPLE_CONDS:
            o_dir = Path(output_dir) / name
            done.append((o_dir / f"{i}.wav").exists())
        if all(done):
            continue

        sig = dataset[i]["signal"]
        results = {
            name: cond(sig, interface).cpu()
            for name, cond in SAMPLE_CONDS.items()
        }

        for name, sig in results.items():
            o_dir = Path(output_dir) / name
            o_dir.mkdir(exist_ok=True, parents=True)

            sig.write(o_dir / f"{i}.wav")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()
