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

            # quantize to 32 steps
            rmsd = torch.round(rmsd * 32)
            return rmsd / 32
        else: 
            return rmsd

@dataclass
class HarmonicChroma:
    hop_length: int
    window_length: int = 4096
    n_chroma: int = 48
    sample_rate: int = 44100


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

        # apply the mask
        chroma = chroma * mask.unsqueeze(-2)

        return chroma[:, 0, :, :-1] # mono only :(  FIX ME!


CONTROLLERS = {
    "rms": RMS, 
    "rmsq": partial(RMS, quantize=True),
    "hchroma": HarmonicChroma,
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
        return {
            k: controller.extract(sig) for k, controller in self.controllers.items()
        }

    def random_mask(self, ctrls: dict[str, Tensor], r: float):
        masks = {}
        for k, ctrl in ctrls.items():
            masks[k] = random_along_time(ctrl, r)
        return masks

    def empty_mask(self, ctrls: dict[str, Tensor]):
        first_key = next(iter(ctrls))
        mask = torch.zeros_like(ctrls[first_key])
        return {k: mask for k in ctrls}
        

def test_controller():
    controller = Sketch2SoundController(
        ctrl_keys=["rms", "hchroma"], 
        hop_length=512, 
        sample_rate=44100
    )

    sig = sn.read_from_file("assets/example.wav")
    sig = sn.read_from_file("/Users/hugo/Downloads/DCS_SE_FullChoir_ScaleUpDown06_A2_DYN.wav")
    sig = sn.excerpt('/Users/hugo/Downloads/(guitarra - hugo mix) bubararu - tambor negro.wav', offset=0, duration=10)
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
    ax2.imshow(ctrls["hchroma"][0][0].cpu().numpy(), aspect='auto', origin='lower')
    # Show the plot
    plt.tight_layout()  # Ensure proper spacing
    plt.show()


if __name__ == "__main__":
    test_controller()
    