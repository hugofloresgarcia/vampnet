
from audiotools import AudioSignal
import torch
from pathlib import Path
import argbind
from tqdm import tqdm
import random

from typing import List

from collections import defaultdict

def coarse2fine_infer(
        signal, 
        model, 
        vqvae, 
        device,
    ):
        output = {}
        w = signal
        w = w.to(device)
        z = vqvae.encode(w.audio_data, w.sample_rate)["codes"]

        model.to(device)
        output["reconstructed"] = model.to_signal(z, vqvae).cpu()

        # make a full mask
        mask = torch.ones_like(z)
        mask[:, :model.n_conditioning_codebooks, :] = 0

        output["sampled"] = model.sample(
            codec=vqvae, 
            time_steps=z.shape[-1], 
            sampling_steps=12, 
            start_tokens=z, 
            mask=mask, 
            temperature=0.85, 
            top_k=None, 
            sample="gumbel", 
            typical_filtering=True, 
            return_signal=True
        ).cpu()

        output["argmax"] = model.sample(
            codec=vqvae, 
            time_steps=z.shape[-1], 
            sampling_steps=1, 
            start_tokens=z, 
            mask=mask, 
            temperature=1.0, 
            top_k=None, 
            sample="argmax", 
            typical_filtering=True, 
            return_signal=True
        ).cpu()

        return output



@argbind.bind(without_prefix=True)
def main(
        sources=[
            "/data/spotdl/audio/val", "/data/spotdl/audio/test"
        ], 
        exp_name="noise_mode",
        model_paths=[
            "runs/c2f-exp-03.22.23/ckpt/mask/epoch=400/vampnet/weights.pth",
            "runs/c2f-exp-03.22.23/ckpt/random/epoch=400/vampnet/weights.pth",
        ],
        model_keys=[
            "mask",
            "random",
        ],
        vqvae_path: str = "runs/codec-ckpt/codec.pth",
        device: str = "cuda",
        output_dir: str = ".",
        max_excerpts: int = 5000,
        duration: float = 3.0,
    ):
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC

    models = {
        k: VampNet.load(p) for k, p in zip(model_keys, model_paths)
    }
    for model in models.values(): 
        model.eval()
    print(f"Loaded {len(models)} models.")

    vqvae = LAC.load(vqvae_path)
    vqvae.to(device)
    vqvae.eval()
    print("Loaded VQVAE.")

    output_dir = Path(output_dir) / f"{exp_name}-samples"

    from audiotools.data.datasets import AudioLoader, AudioDataset

    loader = AudioLoader(sources=sources)
    dataset = AudioDataset(loader, 
        sample_rate=vqvae.sample_rate, 
        duration=duration, 
        n_examples=max_excerpts, 
        without_replacement=True,
    )
    for i in tqdm(range(max_excerpts)):
        sig = dataset[i]["signal"]
        sig.resample(vqvae.sample_rate).normalize(-24).ensure_max_of_audio(1.0)

        for model_key, model in models.items():
            out = coarse2fine_infer(sig, model, vqvae, device)
            out_dir = output_dir / model_key / Path(sig.path_to_file).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, s in out.items():
                s.write(out_dir / f"{k}.wav")
        

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()

