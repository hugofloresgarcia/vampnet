import os
from pathlib import Path
import math
import logging
import PIL


import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch import nn
from torch import Tensor
from torch import tensor as tt
import tqdm
from typing import Optional

import vampnet
from vampnet.util import Timer
from vampnet.modules.transformer import VampNet
from vampnet.mask import *
from vampnet.dsp.signal import cut_to_hop_length, write, trim_to_s
from vampnet.dac.model.dac import DAC
from vampnet.control import Sketch2SoundController
from vampnet.util import first_dict_value, first_dict_key


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
        sig = sn.to_mono(sig)
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

    def get_controls(self,
        insig: sn.Signal,
        controls_periodic_prompt: int = 7,
        controls_drop_amt: float = 0.1,
    ):
        # extract controls and build a mask for them
        ctrls = self.controller.extract(insig)
        ctrl_masks = {}
        # extract onsets, for our onset mask
        onset_idxs = sn.onsets(insig, hop_length=self.codec.hop_length)
        if len(ctrls) > 0:
            rms_key = [k for k in ctrls.keys() if "rms" in k][0]
            ctrl_masks[rms_key] = self.rms_mask(
                ctrls[rms_key], onset_idxs=onset_idxs, 
                periodic_prompt=controls_periodic_prompt, 
                drop_amt=controls_drop_amt
            )
            # use the rms mask for the other controls
            for k in ctrls.keys():
                if k != rms_key:
                    ctrl_masks[k] = ctrl_masks[rms_key]
                    # alternatively, zero it out
                    # ctrl_masks[k] = torch.zeros_like(ctrl_masks["rms"])
        return ctrls, ctrl_masks

    def vamp(self,
        insig: sn.Signal,
        sig_spl: sn.Signal = None,
        seed: int = -1,
        randomize_seed: bool = True,
        controls_periodic_prompt: int = 7,
        controls_drop_amt: float = 0.1,
        codes_periodic_prompt: int = 7,
        upper_codebook_mask: int = 1,
        temperature: float = 1.0,
        mask_temperature: float = 1000.0,
        typical_mass: float = 0.5,
        cfg_scale: float = 5.0,
    ):
        timer = Timer()
            
        if randomize_seed or seed < 0:
            import time
            seed = time.time_ns() % (2**32-1)

        print(f"using seed {seed}")
        sn.seed(seed)

        # preprocess the input signal
        timer.tick("preprocess")
        insig = sn.to_mono(insig)
        inldns = sn.loudness(insig)
        insig = self.preprocess(insig)

        # load the sample (if any)
        if sig_spl is not None:
            # if sig_spl is all zeros
            if torch.all(sig_spl.wav == 0):
                sig_spl = None
                print(f"WARING: sig_sample is all zeros, ignoring")
            else:
                sig_spl = sn.to_mono(sig_spl)
                sig_spl = self.preprocess(sig_spl)
        timer.tock("preprocess")

        timer.tick("controls")
        # extract controls and build a mask for them
        ctrls, ctrl_masks = self.get_controls(insig, 
            controls_periodic_prompt=controls_periodic_prompt, 
            controls_drop_amt=controls_drop_amt
        )
        timer.tock("controls")

        timer.tick("encode")
        # encode the signal
        codes = self.encode(insig.wav)
        timer.tock("encode")

        # make a mask for the codes
        mask = self.build_codes_mask(codes, 
            periodic_prompt=codes_periodic_prompt, 
            upper_codebook_mask=upper_codebook_mask
        )

        timer.tick("prefix")
        if sig_spl is not None:
            # encode the sample
            codes_spl = self.encode(sig_spl.wav)

            # add sample to bundle
            codes, mask, ctrls, ctrl_masks = self.add_sample(
                spl_codes=codes_spl, codes=codes, 
                cmask=mask, ctrls=ctrls, ctrl_masks=ctrl_masks
            )
        timer.tock("prefix")

        # apply the mask
        mcodes = apply_mask(codes, mask, self.vn.mask_token)

        # generate!
        timer.tick("generate")
        print(f"generating with temperature {temperature} and mask temperature {mask_temperature}")
        print(f"typical mass {typical_mass}")
        # breakpoint()
        with torch.autocast(self.device,  dtype=torch.bfloat16):
            gcodes = self.vn.generate(
                codes=mcodes,
                temperature=temperature,
                cfg_scale=cfg_scale,
                mask_temperature=mask_temperature,
                typical_filtering=True,
                typical_mass=typical_mass,
                ctrls=ctrls,
                ctrl_masks=ctrl_masks,
                typical_min_tokens=128,
                sampling_steps=24 if self.vn.mode == "vampnet" else [16, 8, 4, 4],
                causal_weight=0.0,
                debug=False
            )
        timer.tock("generate")

        # remove codes
        if sig_spl is not None:
            gcodes = self.remove_sample(codes_spl, gcodes)

        timer.tick("decode")
        # write the generated signal
        generated_wav = self.decode(gcodes)
        timer.tock("decode")

        return {
            "sig": sn.Signal(generated_wav, insig.sr), 
            "mcodes": gcodes,
            "mask": mask,
            "ctrls": ctrls,
            "ctrl_masks": ctrl_masks
        }

    def preview_input(self, 
        sig: sn.Signal, 
        controls_periodic_prompt: int = 7,
        controls_drop_amt: float = 0.0,
        codes_periodic_prompt: int = 7,
        upper_codebook_mask: int = 1,
    ) -> PIL.Image.Image:
        sig = self.preprocess(sig)
        # get the controls
        ctrls, ctrl_masks = self.get_controls(
            sig, controls_periodic_prompt=controls_periodic_prompt, 
            controls_drop_amt=controls_drop_amt
        )

        # get dummy codes
        codes = torch.randint(0, self.vn.vocab_size, 
            (sig.batch_size, self.vn.n_codebooks, sig.wav.shape[-1] // self.hop_length), 
            device=self.device)
        
        # build a mask
        mask = self.build_codes_mask(codes, 
            periodic_prompt=codes_periodic_prompt, 
            upper_codebook_mask=upper_codebook_mask, 
        )

        # apply the mask
        codes = apply_mask(codes, mask, self.vn.mask_token)

        rmsk = first_dict_key(ctrls)
        rmsmask = ctrl_masks[rmsk]
        # Extract the RMS plot data and apply the mask
        rms_data = ctrls[rmsk][0].cpu().numpy()
        rms_mask_data = rmsmask.cpu().numpy()
        rms_data_masked = rms_data.copy()

        # Fixed dimensions for the output image
        fixed_height = 80  # Number of pixels in height
        fixed_width = 395  # Number of pixels in width
        dpi = 100

        # Define the relative heights for the subplots
        top_height = 4  # Top axis (mask plot) takes 8/9 of the height
        bottom_height = 1  # Bottom axis (RMS plot) takes 1/9 of the heig

        # Set the figure size to match the desired output dimensions
        fig, axes = plt.subplots(2, 1,
            figsize=(fixed_width / dpi, fixed_height / dpi), 
            dpi=dpi, 
            sharex=True, 
            gridspec_kw={"height_ratios": [top_height, bottom_height]}
        )

        # resize the mask
        # mask = mipmap(mask, fixed_width)
        # resize the rms mask
        # rms_mask_data = mipmap(tt(rms_mask_data), fixed_width).cpu().numpy()

        # Display the matrices with interpolation to scale to the fixed size
        ax = axes[0]
        ax.imshow(mask[0].cpu().numpy(), aspect="auto", cmap="Greys", origin="lower", interpolation="nearest")

        rms_data_masked[~rms_mask_data.astype(bool)] = np.nan  # Hide values where the mask is False

        # Plot the RMS data with the masked regions
        rmsax = axes[1]
        # rmsax.plot(rms_data_masked, color="blue")
        rmsax.axis('off')
        rmsax.imshow(rms_data_masked, aspect="auto", cmap="Greys")

        # Remove padding, titles, and axes
        plt.subplots_adjust(hspace=0, wspace=0)
        ax.axis("off")

        plt.tight_layout(pad=0)

        # Export the plot
        outimg = vampnet.util.buffer_plot_and_get(fig, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close()
        return outimg


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




if __name__ == "__main__":
    sig = sn.read_from_file("assets/example.wav")

    from vampnet.train import VampNetTrainer
    ckpt = "hugggof/vampnetv2-tria-d1026-l8-h8-mode-vampnet_rmsq16-latest"
    bundle = VampNetTrainer.from_pretrained(ckpt)

    interface = Interface(
        bundle.codec, 
        bundle.model, 
        bundle.controller
    )
    timer = Timer()
    timer.tick("preview")
    outimg = interface.preview_input(sig)
    timer.tock("preview")
    outimg.save("scratch/input.gif")
