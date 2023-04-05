from pathlib import Path

import argbind
from tqdm import tqdm
import torch

from vampnet.interface import Interface

Interface = argbind.bind(Interface, positional=True)

def baseline(sig, interface):
    return sig

def reconstructed(sig, interface):
    return interface.to_signal(
        interface.encode(sig)
    )

def coarse2fine(sig, interface):
    z = interface.encode(sig)
    z = z[:, :interface.c2f.n_conditioning_codebooks, :]

    z = interface.coarse_to_fine(z)
    return interface.to_signal(z)

def one_codebook(sig, interface):
    z = interface.encode(sig)

    mask = torch.zeros_like(z)
    mask[:, 1:, :] = 1

    zv = interface.coarse_vamp_v2(
        sig, ext_mask=mask,
    )
    zv = interface.coarse_to_fine(zv)  

    return interface.to_signal(zv)

def four_codebooks_downsampled_4x(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, downsample_factor=4
    )
    zv = interface.coarse_to_fine(zv)  
    return interface.to_signal(zv)

def two_codebooks_downsampled_4x(sig, interface):
    z = interface.encode(sig)

    mask = torch.zeros_like(z)
    mask[:, 2:, :] = 1

    zv = interface.coarse_vamp_v2(
        sig, ext_mask=mask, downsample_factor=4
    )
    zv = interface.coarse_to_fine(zv)

    return interface.to_signal(zv)

def four_codebooks_downsampled_8x(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, downsample_factor=8
    )
    zv = interface.coarse_to_fine(zv)  
    return interface.to_signal(zv)





SAMPLE_CONDS ={
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    "one_codebook": one_codebook,
    "four_codebooks_downsampled_4x": four_codebooks_downsampled_4x,
    "two_codebooks_downsampled_4x": two_codebooks_downsampled_4x,
    "four_codebooks_downsampled_8x": four_codebooks_downsampled_8x,
}


@argbind.bind(without_prefix=True)
def main(
        sources=[
            "/data/spotdl/audio/val", "/data/spotdl/audio/test"
        ], 
        output_dir: str = "./samples",
        max_excerpts: int = 5000,
    ):
    interface = Interface()

    output_dir = Path(output_dir) 
    output_dir.mkdir(exist_ok=True, parents=True)

    from audiotools.data.datasets import AudioLoader, AudioDataset

    loader = AudioLoader(sources=sources)
    dataset = AudioDataset(loader, 
        sample_rate=interface.codec.sample_rate, 
        duration=interface.coarse.chunk_size_s, 
        n_examples=max_excerpts, 
        without_replacement=True,
    )

    for i in tqdm(range(max_excerpts)):
        sig = dataset[i]["signal"]
        
        results = {
            name: cond(sig, interface)
            for name, cond in SAMPLE_CONDS.items()
        }

        for name, sig in results.items():
            output_dir = Path(output_dir) / name
            output_dir.mkdir(exist_ok=True, parents=True)

            sig.write(output_dir / f"{i}.wav")

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()
