# ~ basically a copy of audiotools, but torchscript friendly (or so it was) ~
import math
from pathlib import Path
import warnings
import subprocess
from dataclasses import dataclass
import numbers
import time

import torch 
from torch import nn
from torch import Tensor
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import julius

import soundfile as sf
from flatten_dict import flatten
from flatten_dict import unflatten

from .meter import Meter, MIN_LOUDNESS

GAIN_FACTOR = np.log(10) / 20
"""Gain factor for converting between amplitude and decibels."""

@dataclass
class Signal:
    wav: Tensor
    sr: int

    def __post_init__(self):
        assert self.wav.ndim == 3, "Signal must have shape (batch, channels, samples)"
        assert self.wav.shape[-1] > self.wav.shape[-2], "Signal must have more samples than channels! this is just a check. not sure if we'll run into any cases where this is not true."
        
    @property
    def duration(self):
        return self.wav.shape[-1] / self.sr

    @property
    def batch_size(self):
        return self.wav.shape[0]
    
    @property
    def num_channels(self):
        return self.wav.shape[-2]

    @property 
    def num_samples(self):
        return self.wav.shape[-1]

    def to(self, device: str):
        return Signal(self.wav.to(device), self.sr)
    
    def cpu(self):
        return Signal(self.wav.cpu(), self.sr)

    def view(self,):
        return Signal(self.wav, self.sr)

# ~ signal creation utils ~
def batch(
    audio_signals: list[Signal],
    pad_signals: bool = False,
    truncate_signals: bool = False,
    resample: bool = False,
    dim: int = 0,
):
    signal_lengths = [x.num_samples for x in audio_signals]
    sample_rates = [x.sr for x in audio_signals]

    if len(set(sample_rates)) != 1:
        if resample:
            for x in audio_signals:
                x.resample(sample_rates[0])
        else:
            raise RuntimeError(
                f"Not all signals had the same sample rate! Got {sample_rates}. "
                f"All signals must have the same sample rate, or resample must be True. "
            )

    if len(set(signal_lengths)) != 1:
        if pad_signals:
            max_length = max(signal_lengths)
            for x in audio_signals:
                pad_len = max_length - x.num_samples
                padded = zero_pad(x, 0, pad_len)
                x.wav = padded.wav

        elif truncate_signals:
            min_length = min(signal_lengths)
            for x in audio_signals:
                truncated = truncate_samples(x, min_length)
                x.wav = truncated.wav
        else:
            raise RuntimeError(
                f"Not all signals had the same length! Got {signal_lengths}. "
                f"All signals must be the same length, or pad_signals/truncate_signals "
                f"must be True. "
            )
    # Concatenate along the specified dimension (default 0)
    audio_data = torch.cat([x.wav for x in audio_signals], dim=dim)

    batched_signal = Signal(
        audio_data,
        sr=audio_signals[0].sr,
    )
    return batched_signal

# ~ stft ~
def stft(sig: Signal, 
        hop_length: int, window_length: int, 
        center=True
    ):
        nb = sig.wav.shape[0]
        wav = rearrange(sig.wav, 'b c n -> (b c) n')
        stft_d = torch.stft(
            wav, 
            n_fft=window_length, 
            hop_length=hop_length, 
            center=center,
            window=torch.hann_window(window_length).to(wav),
            return_complex=True
        )
        stft_d = rearrange(stft_d, '(b c) f t -> b c f t', b=nb)
        return stft_d

def rms(sig: Signal, window_length: int, **kwargs):
    stft_d = stft(sig, window_length=window_length, **kwargs)
    return rms_from_spec(stft_d, window_length)

def rms_from_spec(spec: Tensor, window_length: int):
    # thank you librosa!
    assert spec.shape[-2] == window_length // 2 + 1, "invalid window length"

    # power spectrogram
    x = torch.abs(spec) ** 2

    # adjust the DC and sr / 2 component
    x[..., 0, :] *= 0.5
    if window_length % 2 == 0:
        x[..., -1, :] *= 0.5
    
    # calculate power
    power = 2 * torch.sum(x, axis=-2, keepdims=False) / window_length ** 2

    rms_d = torch.sqrt(power)
    return rms_d

def onsets(sig: Signal, hop_length: int):
    assert sig.batch_size == 1, "batch size must be 1"
    assert sig.num_channels == 1, "mono signals only"
    import librosa
    onset_frame_idxs = librosa.onset.onset_detect(
        y=sig.wav[0][0].detach().cpu().numpy(), sr=sig.sr, 
        hop_length=hop_length,
        backtrack=True,
    )
    return onset_frame_idxs


# ~ transform ~
def median_filter_1d(x: Tensor, sizes: Tensor) -> Tensor:
    if isinstance(sizes, Tensor):
        sizes = sizes.tolist()
        assert len(sizes) == x.shape[0], "sizes must be the same length as the batch size"
    elif isinstance(sizes, list):
        pass
    else:
        assert isinstance(sizes, int), "sizes must be an int, tensor or list"

    if isinstance(sizes, int):
        if sizes % 2 == 0:
            sizes = sizes + 1
        x = F.pad(x, (sizes // 2, sizes // 2), mode='reflect')
        x = x.unfold(-1, sizes, 1).median(dim=-1)[0]
    else:
        for i, size in enumerate(sizes):
            _x = x[i]
            if size % 2 == 0:
                size = size + 1
            _x = F.pad(_x, (size // 2, size // 2), mode='reflect')
            _x = _x.unfold(-1, size, 1).median(dim=-1)[0]

            x[i] = _x
        
    return x


def pitch_shift(sig: Signal, semitones: int) -> Signal:
    tfm = T.PitchShift(sample_rate=sig.sr, n_steps=semitones)
    return Signal(tfm(sig.wav), sig.sr)

def low_pass(sig: Signal, cutoff: float, zeros: int = 51) -> Signal:
    cutoff = ensure_tensor(cutoff, 2, sig.wav.batch_size).to(sig.wav.device)
    cutoff = cutoff / sig.sr
    filtered = torch.empty_like(sig.wav)

    for i, c in enumerate(cutoff):
        lp_filter = julius.LowPassFilter(c.cpu(), zeros=zeros).to(sig.wav.device)
        filtered[i] = lp_filter(sig.wav[i])
    
    return Signal(filtered, sig.sr)

def high_pass(sig: Signal, cutoff: float, zeros: int = 51) -> Signal:
    cutoff = ensure_tensor(cutoff, 2, sig.wav.batch_size).to(sig.wav.device)
    cutoff = cutoff / sig.sr
    filtered = torch.empty_like(sig.wav)

    for i, c in enumerate(cutoff):
        hp_filter = julius.HighPassFilter(c.cpu(), zeros=zeros).to(sig.wav.device)
        filtered[i] = hp_filter(sig.wav[i])
    
    return Signal(filtered, sig.sr)

def to_mono(sig: Signal) -> Signal:
    """Converts a stereo signal to mono by averaging the channels."""
    wav = sig.wav.mean(dim=-2, keepdim=True)
    return Signal(wav, sig.sr)

def normalize(sig: Signal, db: Tensor | float = -24.0) -> Signal:
    """Normalizes the signal's volume to the specified db, in LUFS.
    This is GPU-compatible, making for very fast loudness normalization.

    Parameters
    ----------
    db : typing.Union[torch.Tensor, np.ndarray, float], optional
        Loudness to normalize to, by default -24.0
    """
    db = ensure_tensor(db).to(sig.wav.device)
    ref_db = loudness(sig)
    gain = db - ref_db
    gain = torch.exp(gain * GAIN_FACTOR)

    wav = sig.wav * gain[:, None, None]
    return Signal(wav, sig.sr)

# ~ analyze ~
def loudness(
    sig: Signal, filter_class: str = "K-weighting", block_size: float = 0.400, **kwargs
) -> Tensor:
    """Calculates loudness using an implementation of ITU-R BS.1770-4.
    Allows control over gating block size and frequency weighting filters for
    additional control. Measure the integrated gated loudness of a signal.

    API is derived from PyLoudnorm, but this implementation is ported to PyTorch
    and is tensorized across batches. When on GPU, an FIR approximation of the IIR
    filters is used to compute loudness for speed.

    Uses the weighting filters and block size defined by the meter
    the integrated loudness is measured based upon the gating algorithm
    defined in the ITU-R BS.1770-4 specification.
    """
    original_length = sig.num_samples
    if sig.duration < 0.5:
        pad_len = int((0.5 - sig.duration) * sig.sr)
        zero_pad(sig, 0, pad_len)

    # create BS.1770 meter
    meter = Meter(
        sig.sr, filter_class=filter_class, block_size=block_size, **kwargs
    )
    meter = meter.to(sig.wav.device)
    # measure loudness
    loudness = meter.integrated_loudness(sig.wav.permute(0, 2, 1))
    truncate_samples(sig, original_length)
    min_loudness = (
        torch.ones_like(loudness, device=loudness.device) * MIN_LOUDNESS
    )
    _loudness = torch.maximum(loudness, min_loudness)

    return _loudness.to(sig.wav.device)

# ~ math ~
def amp2db(x: Tensor) -> Tensor:
    """Converts amplitude to decibels."""
    return 20 * torch.log10(x)

def db2amp(x: Tensor) -> Tensor:
    """Converts decibels to amplitude."""
    return 10 ** (x / 20)

def pow2db(x: Tensor) -> Tensor:
    """Converts power to decibels."""
    return 10 * torch.log10(x)

def db2pow(x: Tensor) -> Tensor:
    """Converts decibels to power."""
    return 10 ** (x / 10)

# ~ sig util ~
@torch.jit.script_if_tracing
def truncate_samples(sig: Signal, original_length: int):
    """Truncates samples to original length."""
    if sig.num_samples > original_length:
        sig.wav = sig.wav[:, :original_length]
    return sig

@torch.jit.script_if_tracing
def zero_pad(sig: Signal, start: int, end: int):
    """Zero pads signal."""
    sig.wav = F.pad(sig.wav, (start, end))
    return sig

@torch.jit.script_if_tracing
def cut_to_hop_length(wav: Tensor, hop_length: int) -> torch.Tensor:
    """Cuts a signal to a multiple of the hop length."""
    length = wav.shape[-1]
    right_cut = length % hop_length
    if right_cut > 0:
        wav = wav[..., :-right_cut]
    return wav

@torch.jit.script_if_tracing
def trim_to_s(sig: Signal, duration: float) -> Tensor:
    """ Trims a signal to a specified duration in seconds."""
    length = int(duration * sig.sr)
    return Signal(sig.wav[..., :length], sig.sr)

def concat(signals: list[Signal]) -> Signal:
    """Concatenates a list of signals along the time axis."""
    first_sig = signals[0]
    assert all([x.sr == first_sig.sr for x in signals]), "All signals must have the same sample rate"
    assert all([x.num_channels == first_sig.num_channels for x in signals]), "All signals must have the same number of channels"
    assert all([x.batch_size == first_sig.batch_size for x in signals]), "All signals must have the same batch size"
    wav = torch.cat([x.wav for x in signals], dim=-1)
    return Signal(wav, signals[0].sr)

def ensure_tensor(
    x: np.ndarray | torch.Tensor | float | int,
    ndim: int = None,
    batch_size: int = None,
):
    """Ensures that the input ``x`` is a tensor of specified
    dimensions and batch size.

    Parameters
    ----------
    x : typing.Union[np.ndarray, torch.Tensor, float, int]
        Data that will become a tensor on its way out.
    ndim : int, optional
        How many dimensions should be in the output, by default None
    batch_size : int, optional
        The batch size of the output, by default None

    Returns
    -------
    torch.Tensor
        Modified version of ``x`` as a tensor.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if ndim is not None:
        assert x.ndim <= ndim
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = x.expand(*shape)
    return x

# ~ data util ~
def prepare_batch(batch: dict | list | torch.Tensor, device: str = "cpu"):
    """Moves items in a batch (typically generated by a DataLoader as a list
    or a dict) to the specified device. This works even if dictionaries
    are nested.

    Parameters
    ----------
    batch : typing.Union[dict, list, torch.Tensor]
        Batch, typically generated by a dataloader, that will be moved to
        the device.
    device : str, optional
        Device to move batch to, by default "cpu"

    Returns
    -------
    typing.Union[dict, list, torch.Tensor]
        Batch with all values moved to the specified device.
    """
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch

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

def collate(list_of_dicts: list, n_splits: int = None):
    """Collates a list of dictionaries (e.g. as returned by a
    dataloader) into a dictionary with batched values. This routine
    uses the default torch collate function for everything
    except Signal objects, which are handled by the batch
    function.

    This function takes n_splits to enable splitting a batch
    into multiple sub-batches for the purposes of gradient accumulation,
    etc.

    Parameters
    ----------
    list_of_dicts : list
        List of dictionaries to be collated.
    n_splits : int
        Number of splits to make when creating the batches (split into
        sub-batches). Useful for things like gradient accumulation.

    """
    batches = []
    list_len = len(list_of_dicts)

    return_list = False if n_splits is None else True
    n_splits = 1 if n_splits is None else n_splits
    n_items = int(math.ceil(list_len / n_splits))

    for i in range(0, list_len, n_items):
        # Flatten the dictionaries to avoid recursion.
        list_of_dicts_ = [flatten(d) for d in list_of_dicts[i : i + n_items]]
        dict_of_lists = {
            k: [dic[k] for dic in list_of_dicts_] for k in list_of_dicts_[0]
        }

        outbatch = {}
        for k, v in dict_of_lists.items():
            if isinstance(v, list):
                if all(isinstance(s, Signal) for s in v):
                    outbatch[k] = batch(v, pad_signals=True)
                else:
                    # Borrow the default collate fn from torch.
                    outbatch[k] = torch.utils.data._utils.collate.default_collate(v)
        batches.append(unflatten(outbatch))

    batches = batches[0] if not return_list else batches
    return batches

# ~ i/o ~
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

def write(sig: Signal, path: Path | str):
    wav = sig.wav
    sr = sig.sr
    if wav[0].abs().max() > 1:
        warnings.warn("Audio amplitude > 1 clipped when saving")

    sf.write(str(path), wav[0].detach().cpu().numpy().T, sr)

def read_from_file(
        path: Path | str, 
        offset: float = 0., 
        duration: float | None = None, 
        device: str = "cpu",
    ) -> Signal:
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

        return Signal(data.to(device), sample_rate)

def resample(sig: Signal, sample_rate: int) -> Signal:
    resampler = T.Resample(orig_freq=sig.sr, new_freq=sample_rate)
    return Signal(resampler(sig.wav), sample_rate)

def excerpt(
    audio_path: str | Path,
    offset: float = None,
    duration: float = None,
    state: np.random.RandomState | int = None,
    **kwargs,
) -> Signal:
    total_duration = fast_get_duration(audio_path)
    if total_duration is None:
        print(f"had to to slow info fall back for {audio_path}")
        _info = info(audio_path)
        total_duration = _info.duration
    try: 
        # Hugo: I think this only works on wav files?
        total_duration = fast_get_duration(audio_path)
    except Exception as e:
        print(e)
        print(f"failed to get fast duration. had to resort to slow info...")
        _info = info(audio_path)
        total_duration = _info.duration

    if duration is None:
        duration = total_duration
        
    state = random_state(state)
    lower_bound = 0 if offset is None else offset
    upper_bound = max(total_duration - duration, 0)
    offset = state.uniform(lower_bound, upper_bound)

    sig = read_from_file(audio_path, offset, duration, **kwargs)
    return sig

# ~ io utils ~
@dataclass
class Info:
    """Shim for torchaudio.info API changes."""

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate

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

# seed
def seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    sig = read_from_file(
        "assets/example.wav"
    )
    rms_d = rms(sig, window_length=2048, hop_length=512)
    print(f"given sig of shape {sig.wav.shape}, rms_d has shape {rms_d.shape}")