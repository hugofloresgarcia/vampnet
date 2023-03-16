import glob
import imp
import os
from pathlib import Path

import argbind
import audiotools
import numpy as np
import pandas as pd
import torch
from flatten_dict import flatten
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

import wav2wav

train = imp.load_source("train", str(Path(__file__).absolute().parent / "train.py"))


@argbind.bind(without_prefix=True)
def evaluate(
    args,
    model_tag: str = "ckpt/best",
    device: str = "cuda",
    exp: str = None,
    overwrite: bool = False,
):
    assert exp is not None

    sisdr_loss = audiotools.metrics.distance.SISDRLoss()
    stft_loss = audiotools.metrics.spectral.MultiScaleSTFTLoss()
    mel_loss = audiotools.metrics.spectral.MelSpectrogramLoss()

    with audiotools.util.chdir(exp):
        vampnet = wav2wav.modules.vampnet.transformer.VampNet.load(
            f"{model_tag}/vampnet/package.pth"
        )
        vampnet = vampnet.to(device)
        if vampnet.cond_dim > 0:
            condnet = wav2wav.modules.condnet.transformer.CondNet.load(
                f"{model_tag}/condnet/package.pth"
            )
            condnet = condnet.to(device)
        else:
            condnet = None

        vqvae = wav2wav.modules.generator.Generator.load(
            f"{model_tag}/vqvae/package.pth"
        )

    _, _, test_data = train.build_datasets(args, vqvae.sample_rate)

    with audiotools.util.chdir(exp):
        datasets = {
            "test": test_data,
        }

        metrics_path = Path(f"{model_tag}/metrics")
        metrics_path.mkdir(parents=True, exist_ok=True)

        for key, dataset in datasets.items():
            csv_path = metrics_path / f"{key}.csv"
            if csv_path.exists() and not overwrite:
                break
            metrics = []
            for i in track(range(len(dataset))):
                # TODO: for coarse2fine
                # grab the signal
                # mask all the codebooks except the conditioning ones
                # and infer
                # then compute metrics
                # for a baseline, just use the coarsest codebook

                try:
                    visqol = audiotools.metrics.quality.visqol(
                        enhanced, clean, "audio"
                    ).item()
                except:
                    visqol = None

                sisdr = sisdr_loss(enhanced, clean)
                stft = stft_loss(enhanced, clean)
                mel = mel_loss(enhanced, clean)

                metrics.append(
                    {
                        "visqol": visqol,
                        "sisdr": sisdr.item(),
                        "stft": stft.item(),
                        "mel": mel.item(),
                        "dataset": key,
                        "condition": exp,
                    }
                )
                print(metrics[-1])

                transform_args = flatten(item["transform_args"], "dot")
                for k, v in transform_args.items():
                    if torch.is_tensor(v):
                        if len(v.shape) == 0:
                            metrics[-1][k] = v.item()

            metrics = pd.DataFrame.from_dict(metrics)
            with open(csv_path, "w") as f:
                metrics.to_csv(f)

        data = summary(model_tag).to_dict()
        metrics = {}
        for k1, v1 in data.items():
            for k2, v2 in v1.items():
                metrics[f"metrics/{k2}/{k1}"] = v2

        # Number of steps to record
        writer = SummaryWriter(log_dir=metrics_path)
        num_steps = 10
        for k, v in metrics.items():
            for i in range(num_steps):
                writer.add_scalar(k, v, i)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate(args)
