import os
from pathlib import Path
import math

import torch
import numpy as np
from audiotools import AudioSignal
import tqdm

from .modules.transformer import VampNet
from .beats import WaveBeat
from .mask import *

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


def signal_concat(
    audio_signals: list,
):
    audio_data = torch.cat([x.audio_data for x in audio_signals], dim=-1)

    return AudioSignal(audio_data, sample_rate=audio_signals[0].sample_rate)

def _load_model(
    ckpt: str, 
    lora_ckpt: str = None,
    device: str = "cpu",
    chunk_size_s: int = 10,
):
    # we need to set strict to False if the model has lora weights to add later
    model = VampNet.load(location=Path(ckpt), map_location="cpu", strict=False)

    # load lora weights if needed
    if lora_ckpt is not None:
        if not Path(lora_ckpt).exists():
            should_cont = input(
                f"lora checkpoint {lora_ckpt} does not exist. continue? (y/n) "
            )
            if should_cont != "y":
                raise Exception("aborting")
        else:
            model.load_state_dict(torch.load(lora_ckpt, map_location="cpu"), strict=False)

    model.to(device)
    model.eval()
    model.chunk_size_s = chunk_size_s
    return model



class Interface(torch.nn.Module):
    def __init__(
        self,
        coarse_ckpt: str = None,
        coarse_lora_ckpt: str = None,
        coarse2fine_ckpt: str = None,
        coarse2fine_lora_ckpt: str = None,
        codec_ckpt: str = None,
        wavebeat_ckpt: str = None,
        device: str = "cpu",
        coarse_chunk_size_s: int =  10, 
        coarse2fine_chunk_size_s: int =  3,
    ):
        super().__init__()
        assert codec_ckpt is not None, "must provide a codec checkpoint"
        self.codec = DAC.load(Path(codec_ckpt))
        self.codec.eval()
        self.codec.to(device)

        assert coarse_ckpt is not None, "must provide a coarse checkpoint"
        self.coarse = _load_model(
            ckpt=coarse_ckpt,
            lora_ckpt=coarse_lora_ckpt,
            device=device,
            chunk_size_s=coarse_chunk_size_s,
        )

        # check if we have a coarse2fine ckpt
        if coarse2fine_ckpt is not None:
            self.c2f = _load_model(
                ckpt=coarse2fine_ckpt,
                lora_ckpt=coarse2fine_lora_ckpt,
                device=device,
                chunk_size_s=coarse2fine_chunk_size_s,
            )
        else:
            self.c2f = None

        if wavebeat_ckpt is not None:
            print(f"loading wavebeat from {wavebeat_ckpt}")
            self.beat_tracker = WaveBeat(wavebeat_ckpt)
            self.beat_tracker.model.to(device)
        else:
            self.beat_tracker = None

        self.device = device

    def lora_load(
        self, 
        coarse_ckpt: str = None,
        c2f_ckpt: str = None,
        full_ckpts: bool = False,
    ):
        if full_ckpts:
            if coarse_ckpt is not None:
                self.coarse = _load_model(
                    ckpt=coarse_ckpt,  
                    device=self.device,
                    chunk_size_s=self.coarse.chunk_size_s,
                )
            if c2f_ckpt is not None:
                self.c2f = _load_model(
                    ckpt=c2f_ckpt,
                    device=self.device,
                    chunk_size_s=self.c2f.chunk_size_s,
                )
        else:
            if coarse_ckpt is not None:
                self.coarse.to("cpu")
                state_dict = torch.load(coarse_ckpt, map_location="cpu")

                self.coarse.load_state_dict(state_dict, strict=False)
                self.coarse.to(self.device)
            if c2f_ckpt is not None:
                self.c2f.to("cpu")
                state_dict = torch.load(c2f_ckpt, map_location="cpu")

                self.c2f.load_state_dict(state_dict, strict=False)
                self.c2f.to(self.device)
        

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

    def to(self, device):
        self.device = device
        self.coarse.to(device)
        self.codec.to(device)

        if self.c2f is not None:
            self.c2f.to(device)

        if self.beat_tracker is not None:
            self.beat_tracker.model.to(device)
        return self

    def to_signal(self, z: torch.Tensor):
        return self.coarse.to_signal(z, self.codec)
    
    def preprocess(self, signal: AudioSignal):
        signal = (
            signal.clone()
            .resample(self.codec.sample_rate)
            .to_mono()
            .normalize(-24)
            .ensure_max_of_audio(1.0)
        )
        return signal
    
    @torch.inference_mode()
    def encode(self, signal: AudioSignal):
        signal = self.preprocess(signal).to(self.device)
        z = self.codec.encode(signal.samples, signal.sample_rate)["codes"]
        return z

    def snap_to_beats(
        self, 
        signal: AudioSignal
    ):
        assert hasattr(self, "beat_tracker"), "No beat tracker loaded"
        beats, downbeats = self.beat_tracker.extract_beats(signal)
        
        # trim the signa around the first beat time
        samples_begin = int(beats[0] * signal.sample_rate )
        samples_end = int(beats[-1] * signal.sample_rate)
        print(beats[0])
        signal = signal.clone().trim(samples_begin, signal.length - samples_end)

        return signal

    def make_beat_mask(self, 
            signal: AudioSignal, 
            before_beat_s: float = 0.1,
            after_beat_s: float = 0.1,
            mask_downbeats: bool = True,
            mask_upbeats: bool = True,
            downbeat_downsample_factor: int = None,
            beat_downsample_factor: int = None,
            dropout: float = 0.0,
            invert: bool = True,
    ):
        """make a beat synced mask. that is, make a mask that 
        places 1s at and around the beat, and 0s everywhere else. 
        """
        assert self.beat_tracker is not None, "No beat tracker loaded"

        # get the beat times
        beats, downbeats = self.beat_tracker.extract_beats(signal)

        # get the beat indices in z
        beats_z, downbeats_z = self.s2t(beats), self.s2t(downbeats)

        # remove downbeats from beats
        beats_z = torch.tensor(beats_z)[~torch.isin(torch.tensor(beats_z), torch.tensor(downbeats_z))]
        beats_z = beats_z.tolist()
        downbeats_z = downbeats_z.tolist()

        # make the mask 
        seq_len = self.s2t(signal.duration)
        mask = torch.zeros(seq_len, device=self.device)
        
        mask_b4 = self.s2t(before_beat_s)
        mask_after = self.s2t(after_beat_s)

        if beat_downsample_factor is not None:
            if beat_downsample_factor < 1:
                raise ValueError("mask_beat_downsample_factor must be >= 1 or None")
        else:
            beat_downsample_factor = 1

        if downbeat_downsample_factor is not None:
            if downbeat_downsample_factor < 1:
                raise ValueError("mask_beat_downsample_factor must be >= 1 or None")
        else:
            downbeat_downsample_factor = 1

        beats_z = beats_z[::beat_downsample_factor]
        downbeats_z = downbeats_z[::downbeat_downsample_factor]
        print(f"beats_z: {len(beats_z)}")
        print(f"downbeats_z: {len(downbeats_z)}")
    
        if mask_upbeats:
            for beat_idx in beats_z:
                _slice = int(beat_idx - mask_b4), int(beat_idx + mask_after)
                num_steps = mask[_slice[0]:_slice[1]].shape[0]
                _m = torch.ones(num_steps, device=self.device)
                _m_mask = torch.bernoulli(_m * (1 - dropout))
                _m = _m * _m_mask.long()
                
                mask[_slice[0]:_slice[1]] = _m

        if mask_downbeats:
            for downbeat_idx in downbeats_z:
                _slice = int(downbeat_idx - mask_b4), int(downbeat_idx + mask_after)
                num_steps = mask[_slice[0]:_slice[1]].shape[0]
                _m = torch.ones(num_steps, device=self.device)
                _m_mask = torch.bernoulli(_m * (1 - dropout))
                _m = _m * _m_mask.long()
                
                mask[_slice[0]:_slice[1]] = _m
        
        mask = mask.clamp(0, 1)
        if invert:
            mask = 1 - mask
        
        mask = mask[None, None, :].bool().long()
        if self.c2f is not None:
            mask = mask.repeat(1, self.c2f.n_codebooks, 1)
        else:
            mask = mask.repeat(1, self.coarse.n_codebooks, 1)
        return mask
        
    def coarse_to_fine(
        self, 
        coarse_z: torch.Tensor,
        **kwargs
    ):
        assert self.c2f is not None, "No coarse2fine model loaded"
        length = coarse_z.shape[-1]
        chunk_len = self.s2t(self.c2f.chunk_size_s)
        n_chunks = math.ceil(coarse_z.shape[-1] / chunk_len)

        # zero pad to chunk_len
        if length % chunk_len != 0:
            pad_len = chunk_len - (length % chunk_len)
            coarse_z = torch.nn.functional.pad(coarse_z, (0, pad_len))

        n_codebooks_to_append = self.c2f.n_codebooks - coarse_z.shape[1]
        if n_codebooks_to_append > 0:
            coarse_z = torch.cat([
                coarse_z,
                torch.zeros(coarse_z.shape[0], n_codebooks_to_append, coarse_z.shape[-1]).long().to(self.device)
            ], dim=1)

        fine_z = []
        for i in range(n_chunks):
            chunk = coarse_z[:, :, i * chunk_len : (i + 1) * chunk_len]
            chunk = self.c2f.generate(
                codec=self.codec,
                time_steps=chunk_len,
                start_tokens=chunk,
                return_signal=False,
                **kwargs
            )
            fine_z.append(chunk)

        fine_z = torch.cat(fine_z, dim=-1)
        return fine_z[:, :, :length].clone()
    
    def coarse_vamp(
        self, 
        z, 
        mask,
        return_mask=False,
        gen_fn=None,
        **kwargs
    ):
        # coarse z
        cz = z[:, : self.coarse.n_codebooks, :].clone()
        assert cz.shape[-1] <= self.s2t(self.coarse.chunk_size_s), f"the sequence of tokens provided must match the one specified in the coarse chunk size, but got {cz.shape[-1]} and {self.s2t(self.coarse.chunk_size_s)}"

        mask = mask[:, : self.coarse.n_codebooks, :]

        cz_masked, mask = apply_mask(cz, mask, self.coarse.mask_token)
        cz_masked = cz_masked[:, : self.coarse.n_codebooks, :]

        gen_fn = gen_fn or self.coarse.generate
        c_vamp = gen_fn(
            codec=self.codec,
            time_steps=cz.shape[-1],
            start_tokens=cz,
            mask=mask, 
            return_signal=False,
            **kwargs
        )

        if return_mask:
            return c_vamp, cz_masked
        
        return c_vamp


if __name__ == "__main__":
    import audiotools as at
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    torch.set_printoptions(threshold=10000)
    at.util.seed(42)

    interface = Interface(
        coarse_ckpt="./models/spotdl/coarse.pth", 
        coarse2fine_ckpt="./models/spotdl/c2f.pth", 
        codec_ckpt="./models/spotdl/codec.pth",
        device="cuda", 
        wavebeat_ckpt="./models/wavebeat.pth"
    )


    sig = at.AudioSignal.zeros(duration=10, sample_rate=44100)

    z = interface.encode(sig)

    # mask = linear_random(z, 1.0)
    # mask = mask_and(
    #     mask, periodic_mask(
    #         z,
    #         32,
    #         1,
    #         random_roll=True
    #     )
    # )

    mask = interface.make_beat_mask(
        sig, 0.0, 0.075
    )
    # mask = dropout(mask, 0.0)
    # mask = codebook_unmask(mask, 0)
    
    breakpoint()
    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        sampling_steps=36,
        temperature=8.0,
        return_mask=True, 
        gen_fn=interface.coarse.generate
    )

    use_coarse2fine = True
    if use_coarse2fine: 
        zv = interface.coarse_to_fine(zv, temperature=0.8)

    mask = interface.to_signal(mask_z).cpu()

    sig = interface.to_signal(zv).cpu()
    print("done")

    sig.write("output3.wav")
    mask.write("mask.wav")
        