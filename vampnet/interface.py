import os
from pathlib import Path
import math
import logging

import torch
from torch import nn
from torch import Tensor
from torch import tensor as tt
import tqdm
from typing import Optional

from vampnet.modules.transformer import VampNet
from vampnet.mask import *
from vampnet.dsp.signal import cut_to_hop_length, write, trim_to_s
from vampnet.dac.model.dac import DAC
from vampnet.control import Sketch2SoundController
from vampnet.util import first_dict_value


# an interface suitable for interfacing with unloop
class Interface(nn.Module):

    def __init__(self,
        codec: DAC, 
        vn: VampNet,
        controller: Optional[Sketch2SoundController] = None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.codec = codec
        self.vn = vn
        self.controller = controller

        self.codec.eval()
        self.vn.eval()

        # compile
        self.sample_rate = self.codec.sample_rate
        self.hop_length = self.codec.hop_length
        self.device = device
        print(f"initialized interface with device {device}")

    def to(self, device):
        self.device = device
        self.codec.to(device)
        self.vn.to(device)
        print(f"interface moved to device {device}")
        return self

    def preprocess(self, sig: sn.Signal) -> sn.Signal:
        sig.wav = sn.cut_to_hop_length(sig.wav, self.hop_length)
        sig = sn.normalize(sig, -16) # TODO: we should refactor this magic number
        sig = sig.to(self.device)
        return sig

    @torch.inference_mode()
    def encode(self, wav):
        nt = wav.shape[-1]
        wav = self.codec.preprocess(wav, self.sample_rate)
        assert nt == wav.shape[-1], f"preprocess function cut off the signal. make sure your input signal is a multiple of hop length"
        z = self.codec.encode(wav)["codes"]
        # chop off, leave only the top  codebooks
        z = z[:, : self.vn.n_codebooks, :]
        return z

    @torch.inference_mode()
    def build_codes_mask(self, 
            codes: Tensor,
            periodic_prompt: Tensor = 13, 
            upper_codebook_mask: Tensor = 3, 
            dropout_amt: Tensor = 0.0,
        ):
        mask = linear_random(codes, 1.0)
        pmask = periodic_mask(codes, periodic_prompt, 1, random_roll=True)
        mask = mask_and(mask, pmask)

        mask = dropout(mask, dropout_amt)

        mask = codebook_mask(mask, upper_codebook_mask, None)
        return mask

    @torch.inference_mode()
    def build_ctrl_masks(self, 
        ctrls: dict[str, Tensor], 
        periodic_prompt: Tensor = 5,
    ):
        ctrl_mask = self.build_codes_mask(
            first_dict_value(ctrls), 
            periodic_prompt=periodic_prompt, 
            upper_codebook_mask=1
        )[:, 0, :]
        ctrl_masks = {
            k: ctrl_mask for k in ctrls.keys()
        }
        return ctrl_masks


    @torch.inference_mode()
    def decode(self, codes):
        return self.codec.decode(self.codec.quantizer.from_codes(codes)[0])


