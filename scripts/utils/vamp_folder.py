from pathlib import Path
import random
from typing import List

import argbind
from tqdm import tqdm
import argbind

from vampnet.interface import Interface
import audiotools as at

Interface = argbind.bind(Interface)

# condition wrapper for printing
def condition(cond):
    def wrapper(sig, interface):
        # print(f"Condition: {cond.__name__}")
        sig = cond(sig, interface)
        # print(f"Condition: {cond.__name__} (done)\n")
        return sig
    return wrapper

@condition
def baseline(sig, interface):
    return interface.preprocess(sig)

@condition
def reconstructed(sig, interface):
    return interface.to_signal(
        interface.encode(sig)
    )

@condition
def coarse2fine(sig, interface):
    z = interface.encode(sig)
    z = z[:, :interface.c2f.n_conditioning_codebooks, :]

    z = interface.coarse_to_fine(z)
    return interface.to_signal(z)

@condition
def coarse2fine_argmax(sig, interface):
    z = interface.encode(sig)
    z = z[:, :interface.c2f.n_conditioning_codebooks, :]

    z = interface.coarse_to_fine(z, 
        sample="argmax", sampling_steps=1, 
        temperature=1.0
    )
    return interface.to_signal(z)

@condition
def one_codebook(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, n_conditioning_codebooks=1
    )
    zv = interface.coarse_to_fine(zv)  

    return interface.to_signal(zv)

@condition
def two_codebooks_downsampled_4x(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, n_conditioning_codebooks=2,
        downsample_factor=4
    )
    zv = interface.coarse_to_fine(zv)

    return interface.to_signal(zv)


def four_codebooks_downsampled(sig, interface, x=12):
    zv = interface.coarse_vamp_v2(
        sig, downsample_factor=12
    )
    zv = interface.coarse_to_fine(zv)  
    return interface.to_signal(zv)


COARSE_SAMPLE_CONDS ={
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    "one_codebook": one_codebook,
    "two_codebooks_downsampled_4x": two_codebooks_downsampled_4x,
    # four codebooks at different downsample factors
    **{
        f"four_codebooks_downsampled_{x}x": lambda sig, interface: four_codebooks_downsampled(sig, interface, x=x)
        for x in [4, 8, 12, 16, 20, 24]
    }

}

C2F_SAMPLE_CONDS = {
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    "coarse2fine_argmax": coarse2fine_argmax,
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

    SAMPLE_CONDS = COARSE_SAMPLE_CONDS if exp_type == "coarse" else C2F_SAMPLE_CONDS


    indices = list(range(max_excerpts))
    random.shuffle(indices)
    for i in tqdm(indices):
        # if all our files are already there, skip
        # done = []
        # for name in SAMPLE_CONDS:
        #     o_dir = Path(output_dir) / name
        #     done.append((o_dir / f"{i}.wav").exists())
        # if all(done):
        #     continue

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
