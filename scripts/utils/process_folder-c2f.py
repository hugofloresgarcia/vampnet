
from audiotools import AudioSignal
import torch
from pathlib import Path
import argbind
from tqdm import tqdm
import random

from collections import defaultdict

def coarse2fine_infer(
        signal, 
        model, 
        vqvae, 
        device,
        signal_window=3, 
        signal_hop=1.5,
        max_excerpts=25, 
    ):
    output = defaultdict(list)

    # split into 3 seconds
    windows = [s for s in signal.clone().windows(signal_window, signal_hop)]
    random.shuffle(windows)
    for w in windows[1:max_excerpts]: # skip the first window since it's mostly zero padded? 
        # batch the signal into chunks of 3
        with torch.no_grad():
            # get codes
            w = w.to(device)
            z = vqvae.encode(w.audio_data, w.sample_rate)["codes"]

            model.to(device)
            output["reconstructed"] = model.to_signal(z, vqvae).cpu()

            # make a full mask
            mask = torch.ones_like(z)
            mask[:, :model.n_conditioning_codebooks, :] = 0

            output["sampled"].append(model.sample(
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
            ).cpu())

            output["argmax"].append(model.sample(
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
            ).cpu())

    return output


@argbind.bind(without_prefix=True)
def main(
        sources=[
            "/home/hugo/data/spotdl/audio/val", "/home/hugo/data/spotdl/audio/test"
        ], 
        audio_ext="mp3",
        exp_name="noise_mode",
        model_paths=[
            "ckpt/mask/best/vampnet/weights.pth",
            "ckpt/random/best/vampnet/weights.pth",
        ],
        model_keys=[
            "noise_mode=mask",
            "noise_mode=random",
        ],
        vqvae_path="ckpt/wav2wav.pth",
        device="cuda",
    ):
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC
    from audiotools.post import audio_zip

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

    audio_dict = defaultdict(list)
    for source in sources:
        print(f"Processing {source}...")
        for path in tqdm(list(Path(source).glob(f"**/*.{audio_ext}"))):
            sig = AudioSignal(path)
            sig.resample(vqvae.sample_rate).normalize(-24).ensure_max_of_audio(1.0)

            for model_key, model in models.items():
                out = coarse2fine_infer(sig, model, vqvae, device)
                for k in out:
                    audio_dict[f"{model_key}-{k}"].extend(out[k])

    audio_zip(audio_dict, f"{exp_name}-results.zip")
    

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()

