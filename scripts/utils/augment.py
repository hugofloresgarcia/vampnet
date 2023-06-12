from pathlib import Path

import audiotools as at
from audiotools import AudioSignal

import argbind
import tqdm


from pedalboard import (
   Compressor, Gain, Chorus, LadderFilter, Phaser, Convolution, Reverb, Pedalboard 
)
from pedalboard.io import AudioFile 

# Read in a whole file, resampling to our desired sample rate:
samplerate = 44100.0
with AudioFile('guitar-input.wav').resampled_to(samplerate) as f:
  audio = f.read(f.frames)

# Make a pretty interesting sounding guitar pedalboard:
board = Pedalboard([
    Compressor(threshold_db=-50, ratio=25),
    Gain(gain_db=30),
    Chorus(),
    LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900),
    Phaser(),
    Convolution("./guitar_amp.wav", 1.0),
    Reverb(room_size=0.25),
])


@argbind.bind(without_prefix=True)
def augment(
    audio_folder: Path,
    dest_folder: Path,
    n_augmentations: int = 10,
):
    """ 
        Augment a folder of audio files by applying audiotools and pedalboard transforms. 

        The dest foler will contain a folder for each of the clean dataset's files. 
        Under each of these folders, there will be a clean file and many augmented files.
    """

    audio_files = at.util.find_audio(audio_folder)

    for audio_file in tqdm.tqdm(audio_files):
        subtree = dest_folder / audio_file.relative_to(audio_folder).parent
        subdir = subtree / audio_file.stem
        subdir.mkdir(parents=True, exist_ok=True)

        # apply pedalboard transforms
        for i in range(n_augmentations):
