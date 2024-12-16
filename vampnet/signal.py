import math
from pathlib import Path
import warnings

import torch 
from torch import nn

import soundfile as sf

import math
import torch
import torch.nn as nn



@torch.jit.script_if_tracing
def cut_to_hop_length(wav: torch.Tensor, hop_length: int) -> torch.Tensor:
    length = wav.shape[-1]
    right_cut = length % hop_length
    if right_cut > 0:
        wav = wav[..., :-right_cut]
    return wav

# ~ i/o ~
def write(wav: torch.Tensor, sr: int, path: Path | str):
    if wav[0].abs().max() > 1:
        warnings.warn("Audio amplitude > 1 clipped when saving")

    sf.write(str(path), wav[0].detach().cpu().numpy().T, sr)
