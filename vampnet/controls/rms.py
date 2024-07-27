
from pathlib import Path
import torch
import numpy as np
import tqdm

from vampnet.signal import Signal as AudioSignal
from dac.utils import load_model 
from dac.model.dac import DAC
import vampnet
from vampnet.controls import STFT_PARAMS


from dataclasses import dataclass
@dataclass
class RMS(vampnet.controls.Control):
    ctrl: torch.Tensor
    name = 'rms'
    hop_size: int = vampnet.HOP_SIZE
    ext: str = ".rms"
    metadata: dict = None

    @classmethod
    def from_signal(cls, sig: AudioSignal, device=vampnet.DEVICE):
        sig.stft_params = STFT_PARAMS
        desc = sig.rms().d
        return cls(desc)
        