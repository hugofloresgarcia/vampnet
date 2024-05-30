from dataclasses import dataclass

import torch

import audiotools as at
import vampnet

@dataclass
class Loudness(vampnet.controls.Control):
    ctrl: torch.Tensor
    name = 'loudness'
    hop_size: int = NotImplemented
    ext = '.ldns'

    @classmethod
    def from_signal(cls, sig: at.AudioSignal):
        raise NotImplementedError

