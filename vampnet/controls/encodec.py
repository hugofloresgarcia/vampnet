
from pathlib import Path
import torch
import numpy as np
import tqdm

from audiotools import AudioSignal
from dac.utils import load_model 
import vampnet

from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# dummy dataset, however you can  this with an dataset on the 🤗 hub or bring your own

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


def _load_codec():
    codec = _load_from_hub()
    assert codec.sample_rate == vampnet.SAMPLE_RATE
    assert codec.hop_length == vampnet.HOP_SIZE
    codec.to(vampnet.DEVICE)
    return codec


def load_codec():
    if not hasattr(vampnet.controls.codec, '_codec'):
        vampnet.controls.codec._codec = _load_codec()
    return vampnet.controls.codec._codec

from dataclasses import dataclass
@dataclass
class DACControl(vampnet.controls.Control):
    ctrl: torch.Tensor
    name = 'dac'
    hop_size: int = vampnet.HOP_SIZE
    ext: str = ".dac"
    metadata: dict = None

    @classmethod
    def from_signal(cls, sig: AudioSignal, device=vampnet.DEVICE):

        from dac.model.base import DACFile
        metadata = {}
        metadata["original_length"] = int(sig.samples.shape[-1])
        metadata["input_db"] = float(sig.ffmpeg_loudness())
        metadata["win_duration"] = vampnet.HOP_SIZE * 10000 / vampnet.SAMPLE_RATE

        sig = sig.normalize(vampnet.LOUD_NORM).ensure_max_of_audio()
        sig.samples = sig.samples.view(-1, 1, sig.samples.shape[-1])

        codes = compress(
                load_codec(), 
                device, 
                sig,
                win_duration=metadata["win_duration"], # like 30s 
        ) # (nt, nch, nc
        codes = codes.permute(2, 1, 0) # (nch, nc, nt)

        return cls(codes, metadata=metadata)