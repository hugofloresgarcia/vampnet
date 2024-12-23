import os
from pathlib import Path
import math
import logging

import torch
from torch import nn
from torch import Tensor
from torch import tensor as tt
from audiotools import AudioSignal
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
        return self.codec.encode(wav)["codes"]

    @torch.inference_mode()
    def build_mask(self, 
            z: Tensor,
            periodic_prompt: Tensor = 7, 
            upper_codebook_mask: Tensor = 3, 
            dropout_amt: Tensor = 0.0,
        ):
        mask = linear_random(z, tt(1.0))
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

        # chop off, leave only the top  codebooks
        print(f"chopping off {self.vn.n_codebooks} codebooks")
        z = z[:, : self.vn.n_codebooks, :]
        mask = mask[:, : self.vn.n_codebooks, :]
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


if __name__ == "__main__":
    import audiotools as at
    import logging
    from torch.export import export
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # torch.set_logging.debugoptions(threshold=10000)
    at.util.seed(42)

    #~~~~ embedded
    print(f"exporting embedded interface")
    ckpt = "/home/hugo/soup/runs/debug/lightning_logs/version_23/checkpoints/epoch=16-step=119232.ckpt"
    codec_ckpt = "/home/hugo/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"

    from scripts.exp.train import VampNetTrainer
    bundle = VampNetTrainer.load_from_checkpoint(ckpt, codec_ckpt=codec_ckpt) 
    codec = bundle.codec
    vn = bundle.model
    eiface = EmbeddedInterface(
        codec=codec,
        vn=vn,
    )
    eiface.eval()
    eiface.to("cpu")

    import vampnet.signal as sn
    sig = sn.read_from_file("assets/example.wav", duration=3.0)
    # handle roughly 10 seconds
    # sig.samples = trim_to_s(sig.samples, sig.sample_rate, 5.0)
    sig.wav = cut_to_hop_length(sig.wav, eiface.hop_length)
    sn.write(sig, "scratch/out_embedded.wav")
    wav = sig.wav.to(eiface.codec.device)

    z = eiface.encode(wav)
    mask = eiface.build_mask(z, 7, 3, 0.3)
    zv = eiface.vamp(z, mask, 1.0, 0.15, 42, 1)

    recons = sn.Signal(eiface.decode(z), eiface.sample_rate)
    out = sn.Signal(eiface.decode(zv), eiface.sample_rate)

    write(recons, "scratch/recons_embedded.wav")
    write(out, "scratch/out_embedded_vamp.wav")

    # https://github.com/Lightning-AI/pytorch-lightning/issues/17517#issuecomment-1528651189
    eiface.vn._trainer = object()

    print(f"exporting embedded interface")
    traced = torch.jit.trace_module(
        mod=eiface, 
        inputs={
            "encode": (wav), 
            "build_mask": (z, tt(7), tt(3), tt(0.3)),
            "vamp": (z, mask, tt(1.0), tt(0.15), tt(42), tt(1)),
            "decode": (zv),
        }
    )
    print("yay! exported without any critical errors.")

    # redo the test with the traced model
    z = traced.encode(wav)
    mask = traced.build_mask(z, tt(0), tt(3), tt(0.3))
    zv = traced.vamp(z, mask, tt(1.0), tt(0.15), tt(42), tt(1))

    recons = sn.Signal(traced.decode(z), eiface.sample_rate)
    out = sn.Signal(traced.decode(zv), eiface.sample_rate)
    write(recons, "scratch/recons_embedded_traced.wav")
    write(out, "scratch/out_embedded_vamp_traced.wav")

    torch.jit.save(traced, "models/vampnet-embedded.pt")

    print(f"expected z shape is {z.shape}")
    print(f"expected mask shape is {mask.shape}")
    print(f"expected zv shape is {zv.shape}")
    print(f"expected wav shape is {sig.wav.shape}")


        