from dataclasses import dataclass
from functools import partial
from typing import Optional

from torch import nn

import vampnet.dsp.signal as sn
from vampnet.dsp.signal import Signal
from vampnet.mask import random_along_time
from torch import Tensor
import torch


class MedianFilterAugment(nn.Module):

    def __init__(self, 
        kernel_size: int, 
        train_min: int = 1, 
        train_max: int = 20,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.train_min = train_min
        self.train_max = train_max

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            sizes = torch.randint(
                self.train_min, 
                self.train_max, 
                size=(x.shape[0],)
            )
        else:
            sizes = self.kernel_size
        # print(f"median filter sizes: {sizes}")
        return sn.median_filter_1d(x, sizes)

class RMS(nn.Module):

    def __init__(self, 
        hop_length, 
        window_length=2048, 
        n_quantize=None, 
        sample_rate=44100, 
        median_filter_size: Optional[int] = None,
        train_median_filter_min=1, 
        train_median_filter_max=15,
    ):
        super().__init__()

        self.hop_length = hop_length
        self.window_length = window_length
        self.n_quantize = n_quantize
        self.sample_rate = sample_rate

        self.mf = MedianFilterAugment(
            kernel_size=median_filter_size, 
            train_min=train_median_filter_min, 
            train_max=train_median_filter_max
        ) if median_filter_size is not None else None
    
    @property
    def dim(self):
        return 1

    def extract(self, sig: Signal) -> Tensor:
        rmsd = sn.rms(sig, 
            window_length=self.window_length, 
            hop_length=self.hop_length, 
        )[:, :, :-1] # TODO: cutting the last frame to match DAC tokens but why :'(
        nb, _, _ = rmsd.shape

        if self.n_quantize is not None:
            # standardize to 0-1
            rmsd = (rmsd - rmsd.min()) / (rmsd.max() - rmsd.min())

            # quantize to 128 steps
            rmsd = torch.round(rmsd * self.n_quantize)
            rmsd =  rmsd / self.n_quantize

        if self.mf is not None:
            rmsd = self.mf(rmsd)
        
        return rmsd



class HarmonicChroma(nn.Module):

    def __init__(self, 
        hop_length: int, window_length: int = 4096,
        n_chroma: int = 48, sample_rate: int = 44100,
        top_n: int = 0
    ):
        super().__init__()
        from torchaudio.prototype.transforms import ChromaScale
        self.hop_length = hop_length
        self.window_length = window_length
        self.n_chroma = n_chroma
        self.sample_rate = sample_rate
        self.top_n = top_n

        # HUGO: this representation, as is,
        # encodes timbre information in the chroma
        # which is not what we want!!!
        # would a median filter help perhaps? 
        self.chroma = ChromaScale(
            sample_rate=self.sample_rate,
            n_freqs=self.window_length // 2 + 1,
            n_chroma=self.n_chroma,
            octwidth=5.0,
        )

    @property
    def dim(self):
        return self.n_chroma

    def extract(self, sig: Signal) -> Tensor:
        from vampnet.dsp.hpss import hpss
        self.chroma.to(sig.wav.device)

        # spectrogram 
        spec = sn.stft(sig, 
            window_length=self.window_length, 
            hop_length=self.hop_length
        )
        # magnitude
        spec = torch.abs(spec)

        # hpss
        spec = hpss(spec, kernel_size=51, hard=True)[0]

        # chroma
        chroma = self.chroma(spec)

        # get the rms of this spec
        rms_d = sn.rms_from_spec(
            spec, window_length=self.window_length
        )

        # convert the rms to db
        rms_d = 10 * torch.log10(rms_d + 1e-7)

        # make a mask based on the rms < -40
        mask = torch.where(rms_d < -40, torch.zeros_like(rms_d), torch.ones_like(rms_d))

        # remove anything below 80 (where the fuck did I get this number from?)
        chroma = torch.where(chroma < 100, torch.zeros_like(chroma), chroma)

        # Get top 2 values and indices along the -2 dimension
        if self.top_n:
            _, topk_indices = torch.topk(chroma, self.top_n, dim=-2)

            # Create a mask for the top 2 values
            topk_mask = torch.zeros_like(chroma).scatter_(-2, topk_indices, 1.0)

            # Retain only the top 2 values
            chroma = chroma * topk_mask

        # apply the mask
        chroma = chroma * mask.unsqueeze(-2)

        # Apply softmax along dim=-2
        if self.top_n > 0:
            chroma = torch.nn.functional.softmax(chroma, dim=-2)

            # mask out any timesteps whose chroma have all equal values (all 0s before softmax)
            # TODO: i did this with chatgpt, there's gott a be a better way
            chroma_mean = chroma.mean(dim=-2, keepdim=True)
            chroma_diff = torch.abs(chroma - chroma_mean)
            equal_mask = torch.all(chroma_diff < 1e-6, dim=-2, keepdim=True)
            
            # Set chroma values to zero for timesteps with all equal values
            chroma = torch.where(equal_mask, torch.zeros_like(chroma), chroma)


        return chroma[:, 0, :, :-1] # mono only :(  FIX ME!


# TODO: try harmonic mel? 

CONTROLLERS = {
    "rms": RMS, 
    "rmsq128": partial(RMS, n_quantize=128),
    "rmsq16": partial(RMS, n_quantize=16),
    "rms-median": partial(RMS, median_filter_size=5),
    "rmsq16-median": partial(RMS, n_quantize=16, median_filter_size=3),
    "hchroma": HarmonicChroma,
    "hchroma-12c-top2": partial(HarmonicChroma, n_chroma=12,  top_n=2), # TODO: refactor me. If this works, this should just be named hchroma. 
    "hchroma-36c-top3": partial(HarmonicChroma, n_chroma=36,  top_n=3) # TODO: refactor me. If this works, this should just be named hchroma.
}
 
class Sketch2SoundController(nn.Module):

    def __init__(
        self,
        ctrl_keys: list[str], 
        hop_length: str, 
        sample_rate: int,
    ):
        super().__init__()

        assert all([k in CONTROLLERS for k in ctrl_keys]), f"got an unsupported control key in {ctrl_keys}!\n  supported: {CONTROLLERS.keys()}"

        self.hop_length = hop_length
        self.ctrl_keys = ctrl_keys
        self.sample_rate = sample_rate

        self.controllers = {
            k: CONTROLLERS[k](hop_length=hop_length, sample_rate=sample_rate)
            for k in self.ctrl_keys
        }

    @property
    def ctrl_dims(self, ) -> dict[str, int]:
        return {
            k: controller.dim for k, controller in self.controllers.items()
        }

    def extract(self, sig: Signal) -> dict[str, Tensor]:
        ctrls = {
            k: controller.extract(sig) for k, controller in self.controllers.items()
        }
        return ctrls

    def random_mask(self, ctrls: dict[str, Tensor], r: float):
        masks = {}
        for k, ctrl in ctrls.items():
            masks[k] = 1-random_along_time(ctrl, r)
        return masks

    def empty_mask(self, ctrls: dict[str, Tensor]):
        first_key = next(iter(ctrls))
        mask = torch.zeros_like(ctrls[first_key])
        return {k: mask for k in ctrls}
        

def test_controller():
    controller = Sketch2SoundController(
        ctrl_keys=["rms-median", "rms", "rmsq128"], 
        hop_length=512, 
        sample_rate=44100
    )
    controller.train()
    # sig = sn.read_from_file("assets/example.wav")
    # sig = sn.read_from_file("/Users/hugo/Downloads/DCS_SE_FullChoir_ScaleUpDown06_A2_DYN.wav")
    # sig = sn.excerpt('/Users/hugo/Downloads/(guitarra - hugo mix) bubararu - tambor negro.wav', offset=0, duration=10)
    sig = sn.read_from_file("assets/voice-prompt.wav")
    ctrls = controller.extract(sig)
    print(f"given sig of shape {sig.wav.shape}, extracted controls: {ctrls}")

    # print the whole thing
    # torch.set_printoptions(profile="full")
    # print(ctrls["hchroma"][0][0][:, 200:210])

    # imshow the chroma
    import matplotlib.pyplot as plt

    # Define relative heights for the subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, 
        sharex=True, 
    )

    # Display the spectrogram on the top
    ax1.imshow(sn.stft(sig, hop_length=512, window_length=2048).abs()[0][0].cpu().log().numpy(), aspect='auto', origin='lower')
    # display rms on the bottom
    ax2.plot(ctrls["rms-median"][0][0])
    ax3.plot(ctrls["rms"][0][0])
    ax4.plot(ctrls["rmsq128"][0][0])

    plt.tight_layout()  # Ensure proper spacing
    plt.savefig("img.png")


if __name__ == "__main__":
    test_controller()