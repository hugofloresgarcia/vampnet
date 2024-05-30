
from audiotools import AudioSignal
from pathlib import Path
from typing import List
import torch
import numpy as np
import json
from dataclasses import dataclass

import vampnet

SUFFIX = ".ctrl"

class Control:
    name: str 
    ctrl: torch.Tensor

    @classmethod
    def from_signal(cls, sig: AudioSignal):
        raise NotImplementedError()
    
    @property
    def num_frames(self):
        return self.ctrl.shape[-1]
    
    @property
    def num_channels(self):
        return self.ctrl.shape[-2]

    @property
    def hop_size(self):
        raise NotImplementedError()


    def normalized(self,):
        raise NotImplementedError


    def validate(self):
        assert self.ctrl.ndim == 3


    def save(self, path: str):
        # use np.savez, pack dict(**vars(self)) into a metadata dict
        num_frames = self.ctrl.shape[-1]
        num_channels = self.ctrl.shape[-2]

        metadata = {
            'name': self.name,
            'num_frames': num_frames,
            'num_channels': num_channels,
        }
        metadata.update(dict(**vars(self)))
        metadata.pop("ctrl")

        # lets make a folder with an extension
        # the ctrl array we'll save as an npz file inside the folder
        # and the metadata we'll save as a json file inside the folder
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        ctrl = self.ctrl.numpy()
        # make the array time first and contiguous
        ctrl = np.ascontiguousarray(ctrl.transpose(2, 0, 1))
        np.savez(path / "ctrl.npz", ctrl=ctrl)

    
    @classmethod
    def load(cls, path: str, offset: int = 0, num_frames: int = None):
        path = Path(path)
        # TODO: this is supposed to be fast lookup for slices, but it takes just as long. 
        # could it be a memory layout thing? 
        ctrl = np.load(path / "ctrl.npz", mmap_mode="r")['ctrl']
        metadata = cls.load_metadata(path)

        num_frames = num_frames or metadata['num_frames']
        # assert ctrl.shape[0] >= offset + num_frames, f" asked for {offset} + {num_frames} frames, but only have {ctrl.shape[-1]} frames"

        metadata.pop("num_frames")
        metadata.pop("num_channels")

        out = torch.from_numpy(ctrl[offset:offset+num_frames, ...]).clone()
        # transpose back out
        out = out.permute(1, 2, 0).contiguous()
        return cls(ctrl=out)


    @classmethod
    def load_metadata(cls, path: str):
        with open(Path(path) / "metadata.json", 'r') as f:
            metadata = json.load(f)
        return metadata
    

def load_control_signal_extractors() -> List[Control]:
    """
    load all control signal extractors that are specified in the vampnet config
    as vampnet.CTRL_KEYS.
    """
    # from vampnet.controls.loudness import Loudness
    from vampnet.controls.codec import DACControl
    extractors = []
    extractors.append(DACControl) # add the codec by default

    for extractor in vampnet.CTRL_KEYS:
        if extractor == "loudness":
            raise NotImplementedError
            # extractors.append(Loudness)
        else:
            raise ValueError(f"unknown control signal extractor: {extractor}")
    return extractors


def test_control_signal():
    # create a dummy control class, put a dummy tensor in it
    # try loading and saving from a temporary folder
    import tempfile
    from pathlib import Path

    @dataclass
    class DummyControl(Control):
        ctrl: torch.Tensor
        name: str = 'dummy'
        meta: str = 'data'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dummy.npz"
        ogctrl = DummyControl(ctrl=torch.randn(2, 3, 4))
        ogctrl.save(path)

        ctrl = DummyControl.load(path, 0, 4)
        assert torch.allclose(ctrl.ctrl, ogctrl.ctrl)

        # try different offsets
        ctrl = DummyControl.load(path, 1, 3)
        assert torch.allclose(ctrl.ctrl, ogctrl.ctrl[..., 1:4])

    # now, let's try with a huge file
    # and compare the time it takes to read a tiny chunk vs a huge one
    import time
    import os

    with tempfile.TemporaryDirectory() as tmpdir:

        # create a huge file
        path = Path(tmpdir) / "dummy.ctrl"
        ogctrl = DummyControl(ctrl=torch.randn(2, 3, 100000000))
        ogctrl.save(path)

        # read a tiny chunk
        start = time.time()
        ctrl = DummyControl.load(path, 0, 1)
        print("time to read tiny chunk:", time.time() - start)

        # read a huge chunk
        start = time.time()
        ctrl = DummyControl.load(path, 0, 10000000)
        print("time to read huge chunk:", time.time() - start)

        # check metadata is what we expect it to be
        metadata = ctrl.load_metadata(path)
        assert metadata['name'] == 'dummy'
        assert metadata['meta'] == 'data'


if __name__ == "__main__":
    test_control_signal()
    print("control signal tests passed")