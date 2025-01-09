from dataclasses import dataclass

import vampnet.signal as sn
from vampnet.signal import Signal
from vampnet.mask import random_along_time
from torch import Tensor
import torch

@dataclass
class RMS:
    hop_length: int
    window_length: int = 2048
    
    @property
    def dim(self):
        return 1

    def extract(self, sig: Signal) -> Tensor:
        return sn.rms(sig, 
            window_length=self.window_length, 
            hop_length=self.hop_length, 
        )[:, :, :-1] # TODO: cutting the last frame to match DAC tokens but why :'(

@dataclass
class RMSQ:
    hop_length: int
    window_length: int = 2048
    
    @property
    def dim(self):
        return 1

    def extract(self, sig: Signal) -> Tensor:
        rmsd = sn.rms(sig,
            window_length=self.window_length, 
            hop_length=self.hop_length, 
        )[:, :, :-1]

        # standardize to 0-1
        rmsd = (rmsd - rmsd.min()) / (rmsd.max() - rmsd.min())

        # quantize to 32 steps
        rmsd = torch.round(rmsd * 32)
        return rmsd / 32

CONTROLLERS = {
    "rms": RMS, 
    "rmsq": RMSQ,
}
class Sketch2SoundController:

    def __init__(
        self,
        ctrl_keys: list[str], 
        hop_length: str, 
    ):
        assert all([k in CONTROLLERS for k in ctrl_keys]), f"got an unsupported control key in {ctrl_keys}!\n  supported: {CONTROLLERS.keys()}"

        self.hop_length = hop_length
        self.ctrl_keys = ctrl_keys

        self.controllers = {
            k: CONTROLLERS[k](hop_length=hop_length)
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
        first_key = next(iter(ctrls))
        mask = random_along_time(ctrls[first_key], r)
        return {k: mask for k in ctrls}

    def empty_mask(self, ctrls: dict[str, Tensor]):
        first_key = next(iter(ctrls))
        mask = torch.zeros_like(ctrls[first_key])
        return {k: mask for k in ctrls}
        

if __name__ == "__main__":
    controller = Sketch2SoundController(
        ctrl_keys=["rms"], 
        hop_length=512
    )

    sig = sn.read_from_file("assets/example.wav")
    ctrls = controller.extract(sig)
    print(f"given sig of shape {sig.wav.shape}, extracted controls: {ctrls}")
    breakpoint()
