from pathlib import Path

import argbind
from tqdm import tqdm
import torch

from vampnet.interface import Interface
import audiotools as at

Interface = argbind.bind(Interface)

# condition wrapper for printing
def condition(cond):
    def wrapper(sig, interface):
        print(f"Condition: {cond.__name__}")
        sig = cond(sig, interface)
        print(f"Condition: {cond.__name__} (done)\n")
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
    z = interface.encode(sig)

    nb, _, nt = z.shape 
    nc = interface.coarse.n_codebooks
    mask = torch.zeros(nb, nc, nt).to(interface.device)
    mask[:, 1:, :] = 1

    zv = interface.coarse_vamp_v2(
        sig, ext_mask=mask,
    )
    zv = interface.coarse_to_fine(zv)  

    return interface.to_signal(zv)

@condition
def four_codebooks_downsampled_4x(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, downsample_factor=4
    )
    zv = interface.coarse_to_fine(zv)  
    return interface.to_signal(zv)

@condition
def two_codebooks_downsampled_4x(sig, interface):
    z = interface.encode(sig)

    nb, _, nt = z.shape 
    nc = interface.coarse.n_codebooks
    mask = torch.zeros(nb, nc, nt).to(interface.device)
    mask[:, 2:, :] = 1

    zv = interface.coarse_vamp_v2(
        sig, ext_mask=mask, downsample_factor=4
    )
    zv = interface.coarse_to_fine(zv)

    return interface.to_signal(zv)

@condition
def four_codebooks_downsampled_8x(sig, interface):
    zv = interface.coarse_vamp_v2(
        sig, downsample_factor=8
    )
    zv = interface.coarse_to_fine(zv)  
    return interface.to_signal(zv)


COARSE_SAMPLE_CONDS ={
    "baseline": baseline,
    "reconstructed": reconstructed,
    "coarse2fine": coarse2fine,
    "one_codebook": one_codebook,
    "four_codebooks_downsampled_4x": four_codebooks_downsampled_4x,
    "two_codebooks_downsampled_4x": two_codebooks_downsampled_4x,
    "four_codebooks_downsampled_8x": four_codebooks_downsampled_8x,
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

    loader = AudioLoader(sources=sources)
    dataset = AudioDataset(loader, 
        sample_rate=interface.codec.sample_rate, 
        duration=interface.coarse.chunk_size_s, 
        n_examples=max_excerpts, 
        without_replacement=True,
    )

    SAMPLE_CONDS = COARSE_SAMPLE_CONDS if exp_type == "coarse" else C2F_SAMPLE_CONDS

    for i in tqdm(range(max_excerpts)):
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
