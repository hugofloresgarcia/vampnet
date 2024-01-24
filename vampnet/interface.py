from audiotools import AudioSignal
import torch
import vampnet.mask as pmask
from vampnet.modules.transformer import VampNet
from dac.utils import load_model as load_dac
import math
import numpy as np

class Interface:

    def __init__(self, 
        vampnet_ckpt = "runs/spotdl-500m-jan19/latest/vampnet/weights.pth",
        codec_ckpt = "models/codec.pth",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        self.vampnet = VampNet.load(vampnet_ckpt).to(self.device)
        self.codec = load_dac(load_path=codec_ckpt).to(self.device)
        self.ckpts = {
            "vampnet": vampnet_ckpt,
            "codec": codec_ckpt,
        }

    def s2t(self, seconds: float):
        """seconds to tokens"""
        if isinstance(seconds, np.ndarray):
            return np.ceil(seconds * self.codec.sample_rate / self.codec.hop_length)
        else:
            return math.ceil(seconds * self.codec.sample_rate / self.codec.hop_length)

    def s2t2s(self, seconds: float):
        """seconds to tokens to seconds"""
        return self.t2s(self.s2t(seconds))

    def t2s(self, tokens: int):
        """tokens to seconds"""
        return tokens * self.codec.hop_length / self.codec.sample_rate
    
    def reload(self, vampnet_ckpt):
        if vampnet_ckpt != self.ckpts["vampnet"]:
            self.vampnet = VampNet.load(vampnet_ckpt).to(self.device)
            self.ckpts["vampnet"] = vampnet_ckpt

    def to(self, device):
        self.device = device
        self.vampnet.to(device)
        self.codec.to(device)
        return self

    def to_signal(self, z: torch.Tensor):
        return self.vampnet.to_signal(z, self.codec, silence_mask=True)

    def preprocess(self, signal: AudioSignal):
        signal = (
            signal.clone()
            .resample(self.codec.sample_rate)
            .to_mono()
            .normalize(-16)
            .ensure_max_of_audio(1.0)
        )
        return signal

    @torch.inference_mode()
    def encode(self, signal: AudioSignal):
        signal = self.preprocess(signal).to(self.device)
        z = self.codec.encode(signal.samples, signal.sample_rate)["codes"]
        return z
    
    def vamp(self, z, mask, return_mask=False, gen_fn=None, **kwargs):
        # coarse z
        cz = z[:, : self.vampnet.n_codebooks, :].clone()
        mask = mask[:, : self.vampnet.n_codebooks, :]

        cz_masked, mask = pmask.apply_mask(cz, mask, self.vampnet.special_tokens["MASK"])
        cz_masked = cz_masked[:, : self.vampnet.n_codebooks, :]

        gen_fn = gen_fn or self.vampnet.generate
        c_vamp = gen_fn(
            codec=self.codec,
            time_steps=cz.shape[-1],
            start_tokens=cz,
            mask=mask,
            return_signal=False,
            **kwargs,
        )

        # add the fine codes back in
        c_vamp = torch.cat(
            [c_vamp, z[:, self.vampnet.n_codebooks :, :]], 
            dim=1
        )

        if return_mask:
            return c_vamp, cz_masked

        return c_vamp

