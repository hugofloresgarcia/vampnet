import os
from pathlib import Path
import math
import logging

import torch
from torch import nn
from torch import Tensor
from torch import tensor as tt
import tqdm

from .modules.transformer import VampNet
from .mask import *
from .signal import cut_to_hop_length, write, trim_to_s

from vampnet.dac.model.dac import DAC


# an interface suitable for interfacing with unloop
class Interface(nn.Module):

    def __init__(self,
        codec: DAC, 
        vn: VampNet,
    ):
        super().__init__()
        self.codec = codec
        self.vn = vn

        self.codec.eval()
        self.vn.eval()

        # compile
        self.sample_rate = self.codec.sample_rate
        self.hop_length = self.codec.hop_length
    

    @torch.inference_mode()
    def encode(self, wav):
        nt = wav.shape[-1]
        wav = self.codec.preprocess(wav, self.sample_rate)
        assert nt == wav.shape[-1], f"preprocess function cut off the signal. make sure your input signal is a multiple of hop length"
        z = self.codec.encode(wav)["codes"]
        # chop off, leave only the top  codebooks
        print(f"chopping off {self.vn.n_codebooks} codebooks")
        z = z[:, : self.vn.n_codebooks, :]
        return z

    @torch.inference_mode()
    def build_mask(self, 
            z: Tensor,
            periodic_prompt: Tensor = 7, 
            upper_codebook_mask: Tensor = 3, 
            dropout_amt: Tensor = 0.0,
        ):
        mask = linear_random(z, 1.0)
        pmask = periodic_mask(z, periodic_prompt, 1, random_roll=True)
        mask = mask_and(mask, pmask)

        mask = dropout(mask, dropout_amt)

        mask = codebook_mask(mask, upper_codebook_mask, None)
        return mask

    @torch.inference_mode()
    def vamp(self, 
        z: Tensor, 
        **kwargs
        ):

        # apply the mask
        z = apply_mask(z, mask, self.vn.mask_token)
        with torch.autocast(z.device.type,  dtype=torch.bfloat16):
            zv = self.vn.generate(
                codes=z,
                **kwargs
            )

        return zv

    @torch.inference_mode()
    def decode(self, z):
        return self.codec.decode(self.codec.quantizer.from_codes(z)[0])


def load_from_trainer_ckpt(ckpt_path=None, codec_ckpt=None):
    from scripts.exp.train import VampNetTrainer
    trainer = VampNetTrainer.load_from_checkpoint(ckpt_path, codec_ckpt=codec_ckpt)
    return Interface(
        codec=trainer.codec,
        vn=trainer.model
    )