from pathlib import Path

import audiotools as at
from audiotools import AudioSignal

import argbind
import tqdm
import torch


from torch_pitch_shift import pitch_shift, get_fast_shifts
from torch_time_stretch import time_stretch, get_fast_stretches

from audiotools.core.util import sample_from_dist


@argbind.bind(without_prefix=True)
def augment(
    audio_folder: Path = None,
    dest_folder: Path = None,
    n_augmentations: int = 10,
):
    """ 
        Augment a folder of audio files by applying audiotools and pedalboard transforms. 

        The dest foler will contain a folder for each of the clean dataset's files. 
        Under each of these folders, there will be a clean file and many augmented files.
    """
    assert audio_folder is not None
    assert dest_folder is not None
    audio_files = at.util.find_audio(audio_folder)

    for audio_file in tqdm.tqdm(audio_files):
        subtree = dest_folder / audio_file.relative_to(audio_folder).parent
        subdir = subtree / audio_file.stem
        subdir.mkdir(parents=True, exist_ok=True)

        src = AudioSignal(audio_file).to("cuda" if torch.cuda.is_available() else "cpu")

        
        for i, chunk in tqdm.tqdm(enumerate(src.windows(10, 10))):
            # apply pedalboard transforms
            for j in range(n_augmentations):
                # pitch shift between -7 and 7 semitones
                import random
                dst = chunk.clone()
                dst.samples = pitch_shift(
                    dst.samples, 
                    shift=random.choice(get_fast_shifts(src.sample_rate, 
                            condition=lambda x: x >= 0.25 and x <= 1.0)), 
                    sample_rate=src.sample_rate
                )
                dst.samples = time_stretch(
                    dst.samples,
                    stretch=random.choice(get_fast_stretches(src.sample_rate, 
                                          condition=lambda x: x >= 0.667 and x <= 1.5, )),
                    sample_rate=src.sample_rate, 
                )

                dst.cpu().write(subdir / f"{i}-{j}.wav")


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        augment()