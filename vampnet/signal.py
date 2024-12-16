import math
from pathlib import Path
import warnings
import subprocess
from dataclasses import dataclass

import torch 
from torch import nn
import numpy as np

import soundfile as sf

import math
import torch
import torch.nn as nn

# ~ util ~
def random_state(seed: int | np.random.RandomState):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )

@torch.jit.script_if_tracing
def cut_to_hop_length(wav: torch.Tensor, hop_length: int) -> torch.Tensor:
    length = wav.shape[-1]
    right_cut = length % hop_length
    if right_cut > 0:
        wav = wav[..., :-right_cut]
    return wav

# ~ i/o ~

@dataclass
class Info:
    """Shim for torchaudio.info API changes."""

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4", ".aiff"]
AUDIO_EXTENSIONS += [x.upper() for x in AUDIO_EXTENSIONS]

def find_audio(folder: str, ext: list[str] = AUDIO_EXTENSIONS):
    """Finds all audio files in a directory recursively.
    Returns a list.

    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    """
    print(f"finding audio in {folder}")
    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        new_files = list(folder.glob(f"**/*{x}"))
        files += new_files
        print(f"found {len(new_files)} files with extension {x}")
    return files



def fast_get_audio_channels(path: str) -> dict:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=channels'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    stdout = stdout.decode().split('\n')[1].split('=')[-1]
    channels = int(stdout)
    return {"path": path, "channels": int(channels)}

def fast_get_duration(path: str) -> float: 

    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=duration'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        duration = float(stdout)
        return duration
    except Exception as e:
        raise e

def fast_get_format(path: str) -> str:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=format_name'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        fmt = stdout
        return fmt
    except Exception as e:
        raise e

def fast_audio_file_info(path:str) -> Info:
    try:
        duration = fast_get_duration(path)
        fmt = fast_get_format(path)
        chan_data = fast_get_audio_channels(path)
        return Info(sample_rate=chan_data['sample_rate'], num_frames=int(duration * chan_data['sample_rate']))
    except Exception as e:
        import warnings
        print(f"Could not process {path}! {e}")
        raise e

def torch_info(audio_path: str):
    try:
        info = torchaudio.info(str(audio_path))
    except:  # pragma: no cover
        info = torchaudio.backend.soundfile_backend.info(str(audio_path))
    return info

def info(audio_path: str):
    """Shim for torchaudio.info to make 0.7.2 API match 0.8.0.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    info = torch_info(audio_path)

    if isinstance(info, tuple):  # pragma: no cover
        signal_info = info[0]
        info = Info(sample_rate=signal_info.rate, num_frames=signal_info.length)
    else:
        info = Info(sample_rate=info.sample_rate, num_frames=info.num_frames)

    return info

def write(wav: torch.Tensor, sr: int, path: Path | str):
    if wav[0].abs().max() > 1:
        warnings.warn("Audio amplitude > 1 clipped when saving")

    sf.write(str(path), wav[0].detach().cpu().numpy().T, sr)

def read_from_file(
        path: Path | str, 
        offset: float, 
        duration: float, 
        device: str = "cpu",
    ):
        import librosa
        try:
            data, sample_rate = librosa.load(
                path,
                offset=offset,
                duration=duration,
                sr=None,
                mono=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {e}")

        data = ensure_tensor(data)
        if data.shape[-1] == 0:
            raise RuntimeError(
                f"Audio file {path} with offset {offset} and duration {duration} is empty!"
            )

        if data.ndim < 2:
            data = data.unsqueeze(0)
        if data.ndim < 3:
            data = data.unsqueeze(0)
        
        self.path_to_file = audio_path
        return self.to(device)


def excerpt(
    audio_path: str | Path,
    offset: float = None,
    duration: float = None,
    state: np.random.RandomState | int = None,
    **kwargs,
):
    total_duration = fast_get_duration(audio_path)
    if total_duration is None:
        print(f"had to to slow info fall back for {audio_path}")
        info = util.info(audio_path)
        total_duration = info.duration
    try: 
        # Hugo: I think this only works on wav files?
        total_duration = util.fast_get_duration(audio_path)

        util_time = time.time() - t0
        # if we took more them 0.5s,log into a debug.txt file
        if util_time > 0.5:
            with open("debug.txt", "a") as f:
                f.write(f"wave_info took {util_time} seconds for {audio_path} \n")
    except Exception as e:
        print(e)
        print(f"failed to get fast duration. had to resort to slow info...")
        info = util.info(audio_path)
        total_duration = info.duration

    info_time = time.time() - t0

    if duration is None:
        duration = total_duration
        
    state = util.random_state(state)
    lower_bound = 0 if offset is None else offset
    upper_bound = max(total_duration - duration, 0)
    offset = state.uniform(lower_bound, upper_bound)

    wav, sr = read_from_file(audio_path, offset, duration, **kwargs)
    return wav, sr