import os
from pathlib import Path
import math
import logging


import numpy as np
import matplotlib.pyplot as plt 
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
        codec.eval()
        vn.eval()
        self.sample_rate = codec.sample_rate
        self.hop_length = codec.hop_length

        self.codec = codec
        self.vn = vn
        self.controller = controller

        self.codec.compile()
        self.vn.compile()

        # compile
        self.device = device
        print(f"initialized interface with device {device}")

    def to(self, device):
        self.device = device
        self.codec.to(device)
        self.vn.to(device)
        print(f"interface moved to device {device}")
        return self

    def preprocess(self, sig: sn.Signal) -> sn.Signal:
        dev = sig.wav.device
        sig = sig.to("cpu") # mps can't support loudness norm
        sig = sn.resample(sig, self.sample_rate)
        sig = sn.normalize(sig, -16) # TODO: we should refactor this magic number
        sig.wav = sn.cut_to_hop_length(sig.wav, self.hop_length)
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

    def add_sample(self, 
        spl_codes: Tensor, 
        codes: Tensor, 
        cmask: Tensor,
        ctrls: Tensor, 
        ctrl_masks: Tensor,
    ):
        """
        update an input bundle (codes, cmask, ctrls, ctrl_masks) with 
        a 'sample prompt' (spl_codes), which is used as a TRIA-like 
        prompt for vampnet. 
        """
        nb = codes.shape[0]
        # if the spl codes have batch 1, repeat them to match the batch size
        if spl_codes.shape[0] == 1:
            print(f"expanding sample codes to batch size {nb}")
            spl_codes = spl_codes.expand(nb, -1, -1)
        assert nb == spl_codes.shape[0], f"batch size mismatch: {nb} != {spl_codes.shape[0]}"
        
        # concatenate the codes
        codes = torch.cat([spl_codes, codes], dim=-1)

        # concatenate controls and control masks
        for ck, ctrl in ctrls.items():
            nc = ctrl.shape[-2]
            ctrls[ck] = torch.cat(
                [torch.zeros(nb, nc, spl_codes.shape[-1]).to(ctrl), ctrl], dim=-1
            )
            ctrl_masks[ck] = torch.cat(
                [torch.zeros(nb, spl_codes.shape[-1]).to(ctrl), ctrl_masks[ck]], dim=-1
            )

        # concatenate the mask
        cmask = torch.cat([empty_mask(spl_codes), cmask], dim=-1)
        
        return codes, cmask, ctrls, ctrl_masks        

    def remove_sample(self,
        spl_codes: Tensor,
        codes: Tensor,
    ):
        """
        remove a sample prompt from the input bundle (codes, cmask, ctrls, ctrl_masks)
        """
        # remove the sample codes
        codes = codes[:, :, spl_codes.shape[-1]:]
        return codes

    @torch.inference_mode()
    def build_codes_mask(self, 
            codes: Tensor,
            periodic_prompt: int = 13, 
            upper_codebook_mask: int = 3, 
            dropout_amt: Tensor = 0.0,
        ):
        mask = linear_random(codes, 1.0)
        pmask = periodic_mask(codes, periodic_prompt, 1, random_roll=False)
        mask = mask_and(mask, pmask)

        assert dropout_amt == 0.0, "dropout is not supported"
        # mask = dropout(mask, dropout_amt)

        mask = codebook_mask(mask, int(upper_codebook_mask), None)
        return mask

    @torch.inference_mode()
    def build_ctrl_mask(self, ctrl: Tensor, periodic_prompt: Tensor = 5):
        return 1-self.build_codes_mask(ctrl, periodic_prompt=periodic_prompt, upper_codebook_mask=1)[:, 0, :]

    def rms_mask(self, 
        rmsd: Tensor, 
        onset_idxs: Tensor, 
        width: int = 5, 
        periodic_prompt=2, 
        drop_amt: float = 0.1
    ):
        mask =  mask_and(
            periodic_mask(rmsd, periodic_prompt, 1, random_roll=False),
            mask_or( # this re-masks the onsets, according to a periodic schedule
                onset_mask(onset_idxs, rmsd, width=width),
                periodic_mask(rmsd, periodic_prompt, 1, random_roll=False),
            )
        ).int()
        # make sure the onset idxs themselves are unmasked
        mask = 1 - mask

        mask[:, :, onset_idxs] = 1
        mask = mask.cpu() # debug
        mask = drop_ones(mask, drop_amt)
        # save mask as txt (ints)
        np.savetxt("scratch/rms_mask.txt", mask[0].cpu().numpy(), fmt='%d')
        mask = mask.to(self.device)
        return mask[:, 0, :]

    def visualize(self, 
        sig: sn.Signal,
        codes: Tensor, 
        mask: Tensor, 
        ctrls: Tensor, 
        ctrl_masks: Tensor,
        out_dir: str = "scratch",
    ):
        num_ctrls = len(ctrls)
        
        # how many axes do we need
        n = 3 + num_ctrls * 2

        fig, axs = plt.subplots(n, 1, figsize=(10, 2*n), sharex=True)
        
        # plot signal
        spec = sn.stft(sig, hop_length=self.hop_length, window_length=self.hop_length*4).abs()
        spec = sn.amp2db(spec)
        axs[0].imshow(spec[0][0].cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")

        # plot codes
        axs[1].imshow(codes[0].cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")
        axs[1].set_title("codes")

        # plot mask
        axs[2].imshow(mask[0].cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")
        axs[2].set_title("mask")

        # plot ctrl and control mask
        for i, (ck, ctrl) in enumerate(ctrls.items()):
            axs[3+i*2].imshow(ctrl[0].cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")
            axs[3+i*2].set_title(ck)

            axs[3+i*2+1].imshow(ctrl_masks[ck].cpu().numpy(), aspect="auto", cmap="viridis", origin="lower")
            axs[3+i*2+1].set_title(f"{ck} mask")
        
        plt.tight_layout()

        # save
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_dir / "bundle.png")
        plt.close()
        return out_dir / "bundle.png"

    @torch.inference_mode()
    def decode(self, codes):
        return self.codec.decode(self.codec.quantizer.from_codes(codes)[0])





