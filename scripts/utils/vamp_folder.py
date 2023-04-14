from pathlib import Path
import random
from typing import List
import tempfile
import subprocess

import argbind
from tqdm import tqdm
import torch

from vampnet.interface import Interface
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

def coarse2fine_argmax(sig, interface):
    z = interface.encode(sig)
    z = z[:, :interface.c2f.n_conditioning_codebooks, :]

    z = interface.coarse_to_fine(z, 
        sample="argmax", sampling_steps=1, 
        temperature=1.0
    )
    return interface.to_signal(z)

class CoarseCond:

    def __init__(self, num_codebooks, downsample_factor):
        self.num_codebooks = num_codebooks
        self.downsample_factor = downsample_factor

    def __call__(self, sig, interface):
        n_conditioning_codebooks = interface.coarse.n_codebooks - self.num_codebooks
        zv = interface.coarse_vamp_v2(sig, 
            n_conditioning_codebooks=n_conditioning_codebooks,
            downsample_factor=self.downsample_factor, 
        )

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

def token_noise(ratio=1.0):
    def wrapper(sig, interface):
        z = interface.encode(sig)
        r = interface.coarse.invgamma(ratio).to(interface.device)
        print(f'adding noise with ratio {ratio}')
        z, mask = interface.coarse.add_noise(
            z, 
            r, 
            noise_mode="random"
        )
        return interface.to_signal(z)
    return wrapper

def mask_ratio_1_step(ratio=1.0):
    def wrapper(sig, interface):
        r = interface.coarse.invgamma(ratio).to(interface.device)
        intensity = 1-r

        zv = interface.coarse_vamp_v2(
            sig, 
            sample='argmax',
            sampling_steps=1, 
            intensity=intensity
        )

        return interface.to_signal(zv)
    return wrapper

def num_sampling_steps(num_steps=1):
    def wrapper(sig, interface):
        zv = interface.coarse_vamp_v2(
            sig, 
            downsample_factor=16,
            sampling_steps=num_steps, 
        )

        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)
    return wrapper

def beat_mask(ctx_time):
    def wrapper(sig, interface):
        beat_mask = interface.make_beat_mask(
            sig,
            before_beat_s=0.0,
            after_beat_s=ctx_time,
            invert=True
        )
        zv = interface.coarse_vamp_v2(
            sig, 
            ext_mask=beat_mask, 
        )

        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)
    return wrapper

def inpaint(ctx_time):
    def wrapper(sig, interface):
        zv = interface.coarse_vamp_v2(
            sig, 
            prefix_dur_s=ctx_time,
            suffix_dur_s=ctx_time,
        )

        zv = interface.coarse_to_fine(zv)
        return interface.to_signal(zv)
    return wrapper

EXP_REGISTRY = {}

EXP_REGISTRY["gen-compression"] = {
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    **{
        f"{n}_codebooks_downsampled_{x}x": CoarseCond(num_codebooks=n, downsample_factor=x)
            for (n, x) in (
                (4, 2), # 4 codebooks, downsampled 2x, 
                (2, 2), # 2 codebooks, downsampled 2x
                (1, None), # 1 codebook, no downsampling
                (4, 4), # 4 codebooks, downsampled 4x
                (1, 2), # 1 codebook, downsampled 2x, 
                (4, 6), # 4 codebooks, downsampled 6x
                (4, 8), # 4 codebooks, downsampled 8x
                (4, 16), # 4 codebooks, downsampled 16x
                (4, 32), # 4 codebooks, downsampled 16x
            )
    }, 

}

EXP_REGISTRY["opus-jazzpop"] = {
    f"opus_{bitrate}": lambda sig, interface: opus(sig, interface, bitrate=bitrate)
    for bitrate in [5620, 1875, 1250, 625]
}

EXP_REGISTRY["opus-spotdl"] = {
    f"opus_{bitrate}": lambda sig, interface: opus(sig, interface, bitrate=bitrate)
    for bitrate in [8036, 2296, 1148, 574]
}

EXP_REGISTRY["opus-baseline"]  = {
    f"opus_{bitrate}": lambda sig, interface: opus(sig, interface, bitrate=bitrate)
    for bitrate in [8000, 12000, 16000]
}

EXP_REGISTRY["c2f"]  = {
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    "coarse2fine_argmax": coarse2fine_argmax,
}

EXP_REGISTRY["token-noise"] = {
    f"token_noise_{r}": token_noise(r)  for r in [0.25, 0.5, 0.75, 1.0]
}

EXP_REGISTRY["mask-ratio"] = {
    "codec": reconstructed,
    **{f"mask_ratio_{r}": mask_ratio_1_step(r)  for r in [0.25, 0.5, 0.75, 0.9]}
}

EXP_REGISTRY["sampling-steps"] = {
    "codec": reconstructed,
    **{f"steps_{n}": num_sampling_steps(n)  for n in [1, 4, 12, 24, 36, 64, 72, 128]},
}

EXP_REGISTRY["baseline"] = {
    "baseline": baseline,
    "codec": reconstructed,
}

EXP_REGISTRY["musical-sampling"] = {
    "baseline": baseline,
    "codec": reconstructed,
    **{f"downsample_{x}x": CoarseCond(4, downsample_factor=x) for x in [16, 32]},
    **{f"beat_mask_{t}": beat_mask(t) for t in [0.075]}, 
    **{f"inpaint_{t}": inpaint(t) for t in [0.5, 1.0,]}, # multiply these by 2 (they go left and right)
}

@argbind.bind(without_prefix=True)
def main(
        sources=[
            "/data/spotdl/audio/val", "/data/spotdl/audio/test"
        ], 
        output_dir: str = "./samples",
        max_excerpts: int = 5000,
        exp_type: str = "coarse", 
        seed: int = 0,
    ):
    at.util.seed(seed)
    interface = Interface()

    output_dir = Path(output_dir) 
    output_dir.mkdir(exist_ok=True, parents=True)

    from audiotools.data.datasets import AudioLoader, AudioDataset

    loader = AudioLoader(sources=sources, shuffle_state=seed)
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
