from pathlib import Path
import os
from functools import partial

from frechet_audio_distance import FrechetAudioDistance
import pandas
import argbind
import torch
from tqdm import tqdm

import audiotools
from audiotools import AudioSignal

@argbind.bind(without_prefix=True)
def eval(
    exp_dir: str = None,
    baseline_key: str = "baseline", 
    audio_ext: str = ".wav",
):
    assert exp_dir is not None
    exp_dir = Path(exp_dir)
    assert exp_dir.exists(), f"exp_dir {exp_dir} does not exist"

    # set up our metrics
    # sisdr_loss = audiotools.metrics.distance.SISDRLoss()
    # stft_loss = audiotools.metrics.spectral.MultiScaleSTFTLoss()
    mel_loss = audiotools.metrics.spectral.MelSpectrogramLoss()
    frechet = FrechetAudioDistance(
        use_pca=False, 
        use_activation=False,
        verbose=True, 
        audio_load_worker=4,
    )
    frechet.model.to("cuda" if torch.cuda.is_available() else "cpu")

    # figure out what conditions we have
    conditions = [d.name for d in exp_dir.iterdir() if d.is_dir()]

    assert baseline_key in conditions, f"baseline_key {baseline_key} not found in {exp_dir}"
    conditions.remove(baseline_key)

    print(f"Found {len(conditions)} conditions in {exp_dir}")
    print(f"conditions: {conditions}")

    baseline_dir = exp_dir / baseline_key 
    baseline_files = sorted(list(baseline_dir.glob(f"*{audio_ext}")), key=lambda x: int(x.stem))

    metrics = []
    for condition in tqdm(conditions):
        cond_dir = exp_dir / condition
        cond_files = sorted(list(cond_dir.glob(f"*{audio_ext}")), key=lambda x: int(x.stem))

        print(f"computing fad for {baseline_dir} and {cond_dir}")
        frechet_score = frechet.score(baseline_dir, cond_dir)

        # make sure we have the same number of files
        num_files = min(len(baseline_files), len(cond_files))
        baseline_files = baseline_files[:num_files]
        cond_files = cond_files[:num_files]
        assert len(list(baseline_files)) == len(list(cond_files)), f"number of files in {baseline_dir} and {cond_dir} do not match. {len(list(baseline_files))} vs {len(list(cond_files))}"

        def process(baseline_file, cond_file):
            # make sure the files match (same name)
            assert baseline_file.stem == cond_file.stem, f"baseline file {baseline_file} and cond file {cond_file} do not match"

            # load the files
            baseline_sig = AudioSignal(str(baseline_file))
            cond_sig = AudioSignal(str(cond_file))

            cond_sig.resample(baseline_sig.sample_rate)
            cond_sig.truncate_samples(baseline_sig.length)

            # if our condition is inpainting, we need to trim the conditioning off
            if "inpaint" in condition:
                ctx_amt = float(condition.split("_")[-1])
                ctx_samples = int(ctx_amt * baseline_sig.sample_rate)
                print(f"found inpainting condition. trimming off {ctx_samples} samples from {cond_file} and {baseline_file}")
                cond_sig.trim(ctx_samples, ctx_samples)
                baseline_sig.trim(ctx_samples, ctx_samples)

            return {
                # "sisdr": -sisdr_loss(baseline_sig, cond_sig).item(),
                # "stft": stft_loss(baseline_sig, cond_sig).item(),
                "mel": mel_loss(baseline_sig, cond_sig).item(),
                "frechet": frechet_score,
                # "visqol": vsq,
                "condition": condition,
                "file": baseline_file.stem,
            }

        print(f"processing {len(baseline_files)} files in {baseline_dir} and {cond_dir}")
        metrics.extend(tqdm(map(process, baseline_files, cond_files), total=len(baseline_files)))

    metric_keys = [k for k in metrics[0].keys() if k not in ("condition", "file")]


    for mk in metric_keys:
        stat = pandas.DataFrame(metrics)
        stat = stat.groupby(['condition'])[mk].agg(['mean', 'count', 'std'])
        stat.to_csv(exp_dir / f"stats-{mk}.csv")

    df = pandas.DataFrame(metrics)
    df.to_csv(exp_dir / "metrics-all.csv", index=False)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        eval()