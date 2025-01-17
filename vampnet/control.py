from dataclasses import dataclass
from functools import partial

import vampnet.dsp.signal as sn
from vampnet.dsp.signal import Signal
from vampnet.mask import random_along_time
from torch import Tensor
import torch

@dataclass
class RMS:
    hop_length: int
    window_length: int = 2048
    quantize: bool = False
    sample_rate: int = 44100 # UNUSED
    
    @property
    def dim(self):
        return 1

    def extract(self, sig: Signal) -> Tensor:
        rmsd = sn.rms(sig, 
            window_length=self.window_length, 
            hop_length=self.hop_length, 
        )[:, :, :-1] # TODO: cutting the last frame to match DAC tokens but why :'(

        if self.quantize:
            # standardize to 0-1
            rmsd = (rmsd - rmsd.min()) / (rmsd.max() - rmsd.min())

            # quantize to 128 steps
            rmsd = torch.round(rmsd * 128)
            return rmsd / 128
        else: 
            return rmsd

@dataclass
class HarmonicChroma:
    hop_length: int
    window_length: int = 4096
    n_chroma: int = 48
    sample_rate: int = 44100
    top_n: int = 2

    # HUGO: this representation, as is,
    # encodes timbre information in the chroma
    # which is not what we want!!!
    # would a median filter help perhaps? 

    def __post_init__(self):
        from torchaudio.prototype.transforms import ChromaScale
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
    "rmsq128": partial(RMS, quantize=True),
    "hchroma": HarmonicChroma,
    "hchroma-12c-top2": partial(HarmonicChroma, n_chroma=12,  top_n=2), # TODO: refactor me. If this works, this should just be named hchroma. 
    "hchroma-36c-top3": partial(HarmonicChroma, n_chroma=36,  top_n=3) # TODO: refactor me. If this works, this should just be named hchroma.
}

class Sketch2SoundController:

    def __init__(
        self,
        ctrl_keys: list[str], 
        hop_length: str, 
        sample_rate: int,
    ):
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
        ctrl_keys=["rms", "hchroma-12c-top2"], 
        hop_length=512, 
        sample_rate=44100
    )

    sig = sn.read_from_file("assets/example.wav")
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
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        sharex=True, 
    )

    # Display the spectrogram on the top
    ax1.imshow(sn.stft(sig, hop_length=512, window_length=2048).abs()[0][0].cpu().numpy(), aspect='auto', origin='lower')
    # Display the chroma on the bottom
    ax2.imshow(ctrls["hchroma-12c-top2"][0].cpu().numpy(), aspect='auto', origin='lower')
    # Show the plot
    plt.tight_layout()  # Ensure proper spacing
    plt.show()


if __name__ == "__main__":
    test_controller()
    