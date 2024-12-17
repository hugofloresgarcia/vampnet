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


# an interface suitable for tracing 
class EmbeddedInterface(nn.Module):

    def __init__(self,
        codec: DAC, 
        coarse: VampNet, 
        chunk_size_s: int = 10,
    ):
        super().__init__()
        self.codec = codec
        self.coarse = coarse

        self.register_buffer("sample_rate", tt(self.codec.sample_rate))
        self.register_buffer("hop_length", tt(self.codec.hop_length))

    @torch.inference_mode()
    def encode(self, wav):
        nt = wav.shape[-1]
        wav = self.codec.preprocess(wav, self.sample_rate)
        assert nt == wav.shape[-1], f"preprocess function cut off the signal. make sure your input signal is a multiple of hop length"
        return self.codec.encode(wav)["codes"]

    @torch.inference_mode()
    def build_mask(self, 
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
        mask: Tensor,
        temperature: Tensor = 1.0,
        typical_mass: Tensor = 0.0,
        typical_min_tokens: Tensor = 0,
        seed: Tensor = 42,
        ):

        # chop off, leave only the top  codebooks
        print(f"chopping off {self.coarse.n_codebooks} codebooks")
        z = z[:, : self.coarse.n_codebooks, :]
        mask = mask[:, : self.coarse.n_codebooks, :]
        # apply the mask
        z = apply_mask(z, mask, self.coarse.mask_token)
        with torch.autocast(z.device.type,  dtype=torch.bfloat16):
            zv = self.coarse.generate(
                codes=z,
                temperature=temperature,
                typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
                seed=seed,
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

    interface = Interface(
        coarse_ckpt="./models/vampnet/coarse.pth", 
        coarse2fine_ckpt="./models/vampnet/c2f.pth", 
        codec_ckpt="./models/vampnet/codec.pth",
        device="cuda", 
        compile=False
    )

    sig = at.AudioSignal('assets/example.wav')

    z = interface.encode(sig)

    for m in interface.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)


    mask = interface.build_mask(
        z=z,
        sig=sig,
        rand_mask_intensity=1.0,
        prefix_s=0.0,
        suffix_s=0.0,
        periodic_prompt=7,
        periodic_prompt_width=1,
        onset_mask_width=5, 
        _dropout=0.0,
        upper_codebook_mask=3,
    )

    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        return_mask=True, 
        gen_fn=interface.coarse.generate
    )

    use_coarse2fine = True
    if use_coarse2fine: 
        zv = interface.coarse_to_fine(zv, mask=mask)
    
    sig = interface.decode(zv).cpu()
    mask = interface.decode(mask_z).cpu()
    recons = interface.decode(z).cpu()

    sig.write("scratch/out.wav")
    mask.write("scratch/mask.wav")
    recons.write("scratch/recons.wav")

    logging.debug("done")

    #~~~~ embedded
    print(f"exporting embedded interface")
    # move stuff to cpu before exporting
    z = z.cpu()
    mask = mask.cpu()
    zv = zv.cpu()
    interface.codec.to("cpu")
    interface.coarse.to("cpu")
    sig = sig.to("cpu")

    eiface = EmbeddedInterface(
        codec=interface.codec, 
        coarse=interface.coarse,
    )
    eiface.eval()

    # handle roughly 10 seconds
    sig.samples = trim_to_s(sig.samples, sig.sample_rate, 5.0)
    sig.samples = cut_to_hop_length(sig.samples, eiface.hop_length)
    sig.write("scratch/out_embedded.wav")
    wav = sig.samples.to(eiface.codec.device)

    z = eiface.encode(wav)
    mask = eiface.build_mask(7, 3, 0.3)
    zv = eiface.vamp(z, mask, 1.0, 0.15, 42, 1)

    recons = eiface.decode(z)
    out = eiface.decode(zv)

    write(recons, eiface.sample_rate, "scratch/recons_embedded.wav")
    write(out, eiface.sample_rate, "scratch/out_embedded_vamp.wav")

    print(f"exporting embedded interface")
    traced = torch.jit.trace_module(
        mod=eiface, 
        inputs={
            "encode": (wav), 
            "build_mask": (tt(7), tt(3), tt(0.3)),
            "vamp": (z, mask, tt(1.0), tt(0.15), tt(42), tt(1)),
            "decode": (zv),
        }
    )
    print("yay! exported without any critical errors.")

    # redo the test with the traced model
    z = traced.encode(wav)
    mask = traced.build_mask(tt(7), tt(3), tt(0.3))
    zv = traced.vamp(z, mask, tt(1.0), tt(0.15), tt(42), tt(1))
    out = traced.decode(zv)

    write(recons, eiface.sample_rate, "scratch/recons_embedded_traced.wav")
    write(out, eiface.sample_rate, "scratch/out_embedded_vamp_traced.wav")

    torch.jit.save(traced, "models/vampnet-embedded.pt")

    print(f"expected z shape is {z.shape}")
    print(f"expected mask shape is {mask.shape}")
    print(f"expected zv shape is {zv.shape}")
    print(f"expected wav shape is {sig.samples.shape}")


        