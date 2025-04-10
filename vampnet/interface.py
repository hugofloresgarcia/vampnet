import os
from pathlib import Path
import math
import logging

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
        wavebeat_ckpt: str = "./models/vampnet/wavebeat.pth",
        device: str = "cpu",
        coarse_chunk_size_s: int =  10, 
        coarse2fine_chunk_size_s: int =  3,
        compile=True,
    ):
        super().__init__()
        assert codec_ckpt is not None, "must provide a codec checkpoint"
        self.codec = DAC.load(Path(codec_ckpt))
        self.codec.eval()
        self.codec.to(device)
        self.codec_path = Path(codec_ckpt)

        assert coarse_ckpt is not None, "must provide a coarse checkpoint"
        self.coarse = _load_model(
            ckpt=coarse_ckpt,
            lora_ckpt=coarse_lora_ckpt,
            device=device,
            chunk_size_s=coarse_chunk_size_s,
        )
        self.coarse_path = Path(coarse_ckpt)

        # check if we have a coarse2fine ckpt
        if coarse2fine_ckpt is not None:
            self.c2f_path = Path(coarse2fine_ckpt)
            self.c2f = _load_model(
                ckpt=coarse2fine_ckpt,
                lora_ckpt=coarse2fine_lora_ckpt,
                device=device,
                chunk_size_s=coarse2fine_chunk_size_s,
            )
        else:
            self.c2f_path = None
            self.c2f = None

        if wavebeat_ckpt is not None:
            logging.debug(f"loading wavebeat from {wavebeat_ckpt}")
            self.beat_tracker = WaveBeat(wavebeat_ckpt, device=device)
            self.beat_tracker.model.to(device)
        else:
            self.beat_tracker = None

        self.device = device
        self.loudness = -24.0

        if compile:
            logging.debug(f"compiling models")
            self.coarse = torch.compile(self.coarse)
            if self.c2f is not None:
                self.c2f = torch.compile(self.c2f)
            self.codec = torch.compile(self.codec)


    @classmethod
    def default(cls):
        from . import download_codec, download_default
        print(f"loading default vampnet")
        codec_path = download_codec()
        coarse_path, c2f_path = download_default()
    
        return Interface(
            coarse_ckpt=coarse_path,
            coarse2fine_ckpt=c2f_path,
            codec_ckpt=codec_path,
        )

    @classmethod
    def available_models(cls):
        from . import list_finetuned
        return list_finetuned() + ["default"]


    def load_finetuned(self, name: str):
        assert name in self.available_models(), f"{name} is not a valid model name"
        from . import download_finetuned, download_default
        if name == "default":
            coarse_path, c2f_path = download_default()
        else:
            coarse_path, c2f_path = download_finetuned(name)
        self.reload(
            coarse_ckpt=coarse_path,
            c2f_ckpt=c2f_path,
        )

    def reload(
        self, 
        coarse_ckpt: str = None,
        c2f_ckpt: str = None,
    ):
        if coarse_ckpt is not None:
            # check if we already loaded, if so, don't reload
            if self.coarse_path == Path(coarse_ckpt):
                logging.debug(f"already loaded {coarse_ckpt}")
            else:
                self.coarse = _load_model(
                    ckpt=coarse_ckpt,  
                    device=self.device,
                    chunk_size_s=self.coarse.chunk_size_s,
                )
                self.coarse_path = Path(coarse_ckpt)
                logging.debug(f"loaded {coarse_ckpt}")

        if c2f_ckpt is not None:
            if self.c2f_path == Path(c2f_ckpt):
                logging.debug(f"already loaded {c2f_ckpt}")
            else:
                self.c2f = _load_model(
                    ckpt=c2f_ckpt,
                    device=self.device,
                    chunk_size_s=self.c2f.chunk_size_s,
                )
                self.c2f_path = Path(c2f_ckpt)
                logging.debug(f"loaded {c2f_ckpt}")
        
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

    def decode(self, z: torch.Tensor):
        return self.coarse.decode(z, self.codec)
    
    def _preprocess(self, signal: AudioSignal):
        signal = (
            signal.clone()
            .resample(self.codec.sample_rate)
            .to_mono()
            .normalize(self.loudness)
            .ensure_max_of_audio(1.0)
        )
        logging.debug(f"length before codec preproc: {signal.samples.shape}")
        signal.samples, length = self.codec.preprocess(signal.samples, signal.sample_rate)
        logging.debug(f"length after codec preproc: {signal.samples.shape}")
        return signal
    
    @torch.inference_mode()
    def encode(self, signal: AudioSignal):
        signal = signal.to(self.device)
        signal = self._preprocess(signal)
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
        logging.debug(beats[0])
        signal = signal.clone().trim(samples_begin, signal.length - samples_end)

        return signal

    def make_beat_mask(self, 
            signal: AudioSignal, 
            before_beat_s: float = 0.0,
            after_beat_s: float = 0.02,
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
        logging.debug(f"beats_z: {len(beats_z)}")
        logging.debug(f"downbeats_z: {len(downbeats_z)}")
    
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
    
    def set_chunk_size(self, chunk_size_s: float):
        self.coarse.chunk_size_s = chunk_size_s
        
    @torch.inference_mode()
    def coarse_to_fine(
        self, 
        z: torch.Tensor,
        mask: torch.Tensor = None,
        return_mask: bool = False,
        **kwargs
    ):
        assert self.c2f is not None, "No coarse2fine model loaded"
        length = z.shape[-1]
        chunk_len = self.s2t(self.c2f.chunk_size_s)
        n_chunks = math.ceil(z.shape[-1] / chunk_len)

        # zero pad to chunk_len
        if length % chunk_len != 0:
            pad_len = chunk_len - (length % chunk_len)
            z = torch.nn.functional.pad(z, (0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=1) if mask is not None else None

        n_codebooks_to_append = self.c2f.n_codebooks - z.shape[1]
        if n_codebooks_to_append > 0:
            z = torch.cat([
                z,
                torch.zeros(z.shape[0], n_codebooks_to_append, z.shape[-1]).long().to(self.device)
            ], dim=1)
            logging.debug(f"appended {n_codebooks_to_append} codebooks to z")

        # set the mask to 0 for all conditioning codebooks
        if mask is not None:
            mask = mask.clone()
            mask[:, :self.c2f.n_conditioning_codebooks, :] = 0

        fine_z = []
        for i in range(n_chunks):
            chunk = z[:, :, i * chunk_len : (i + 1) * chunk_len]
            mask_chunk = mask[:, :, i * chunk_len : (i + 1) * chunk_len] if mask is not None else None
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.c2f.generate(
                    codec=self.codec,
                    time_steps=chunk_len,
                    start_tokens=chunk,
                    return_signal=False,
                    mask=mask_chunk,
                    cfg_guidance=None,
                    **kwargs
                )
                fine_z.append(chunk)

        fine_z = torch.cat(fine_z, dim=-1)
        if return_mask:
            return fine_z[:, :, :length].clone(), apply_mask(fine_z, mask, self.c2f.mask_token)[0][:, :, :length].clone()
        
        return fine_z[:, :, :length].clone()
    
    @torch.inference_mode()
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
        mask = mask[:, : self.coarse.n_codebooks, :]
        # assert cz.shape[-1] <= self.s2t(self.coarse.chunk_size_s), f"the sequence of tokens provided must match the one specified in the coarse chunk size, but got {cz.shape[-1]} and {self.s2t(self.coarse.chunk_size_s)}"

        # cut into chunks, keep the last chunk separate if it's too small
        chunk_len = self.s2t(self.coarse.chunk_size_s)
        n_chunks = math.ceil(cz.shape[-1] / chunk_len)
        last_chunk_len = cz.shape[-1] % chunk_len

        cz_chunks = []
        mask_chunks = []
        for i in range(n_chunks):
            chunk = cz[:, :, i * chunk_len : (i + 1) * chunk_len]
            mask_chunk = mask[:, :, i * chunk_len : (i + 1) * chunk_len]

            # make sure that the very first and last timestep of each chunk is 0 so that we don't get a weird
            # discontinuity when we stitch the chunks back together
            # only if there's already a 0 somewhere in the chunk
            if torch.any(mask_chunk == 0):
                mask_chunk = mask_chunk.clone()
                mask_chunk[:, :, 0] = 0
                mask_chunk[:, :, -1] = 0

            cz_chunks.append(chunk)
            mask_chunks.append(mask_chunk)

        # now vamp each chunk
        cz_masked_chunks = []
        cz_vamped_chunks = []
        for chunk, mask_chunk in zip(cz_chunks, mask_chunks):
            cz_masked_chunk, mask_chunk = apply_mask(chunk, mask_chunk, self.coarse.mask_token)
            cz_masked_chunk = cz_masked_chunk[:, : self.coarse.n_codebooks, :]
            cz_masked_chunks.append(cz_masked_chunk)
            

            gen_fn = gen_fn or self.coarse.generate
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c_vamp_chunk = gen_fn(
                    codec=self.codec,
                    time_steps=chunk_len,
                    start_tokens=cz_masked_chunk,
                    return_signal=False,
                    mask=mask_chunk,
                    **kwargs
                )
                cz_vamped_chunks.append(c_vamp_chunk)
        
        # stitch the chunks back together
        cz_masked = torch.cat(cz_masked_chunks, dim=-1)
        c_vamp = torch.cat(cz_vamped_chunks, dim=-1)

        # add the fine codes back in
        c_vamp = torch.cat(
            [c_vamp, z[:, self.coarse.n_codebooks :, :]], 
            dim=1
        )

        if return_mask:
            return c_vamp, cz_masked
        
        return c_vamp
    
    def build_mask(self, 
        z: torch.Tensor,
        sig: AudioSignal = None,
        rand_mask_intensity: float = 1.0,
        prefix_s: float = 0.0,
        suffix_s: float = 0.0,
        periodic_prompt: int = 7,
        periodic_prompt_width: int = 1,
        onset_mask_width: int = 0, 
        _dropout: float = 0.0,
        upper_codebook_mask: int = 3,
        ncc: int = 0,
    ):
        mask = linear_random(z, rand_mask_intensity)
        mask = mask_and(
            mask,
            inpaint(z, self.s2t(prefix_s), self.s2t(suffix_s)),
        )

        pmask = periodic_mask(z, periodic_prompt, periodic_prompt_width, random_roll=True)
        mask = mask_and(mask, pmask)

        if onset_mask_width > 0:
            assert sig is not None, f"must provide a signal to use onset mask"
            mask = mask_and(
                mask, onset_mask(
                    sig, z, self, 
                    width=onset_mask_width
                )
            )

        mask = dropout(mask, _dropout)
        mask = codebook_unmask(mask, ncc)

        mask = codebook_mask(mask, int(upper_codebook_mask), None)
        return mask

    def vamp(
        self, 
        codes: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int = 1,
        feedback_steps: int = 1,
        time_stretch_factor: int = 1,
        return_mask: bool = False,
        **kwargs,
    ):
        z = codes

        # expand z to batch size
        z = z.expand(batch_size, -1, -1)
        mask = mask.expand(batch_size, -1, -1)

        # stretch mask and z to match the time stretch factor
        # we'll add (stretch_factor - 1) mask tokens in between each timestep of z
        # and we'll make the mask 1 in all the new slots we added
        if time_stretch_factor > 1:
            z = z.repeat_interleave(time_stretch_factor, dim=-1)
            mask = mask.repeat_interleave(time_stretch_factor, dim=-1)
            added_mask = torch.ones_like(mask)
            added_mask[:, :, ::time_stretch_factor] = 0
            mask = mask.bool() | added_mask.bool()
            mask = mask.long()
            
        # the forward pass
        logging.debug(z.shape)
        logging.debug("coarse!")
        zv = z
        for i in range(feedback_steps):
            zv, mask_z = self.coarse_vamp(
                zv, 
                mask=mask,
                return_mask=True, 
                **kwargs)
            # roll the mask around a random amount
            mask_z = mask_z.roll(
                shifts=(i + 1) % feedback_steps, 
                dims=-1
            )


        # add the top codebooks back in
        if zv.shape[1] < z.shape[1]:
            logging.debug(f"adding {z.shape[1] - zv.shape[1]} codebooks back in")
            zv = torch.cat(
                [zv, z[:, self.coarse.n_codebooks :, :]], 
                dim=1
            )

        # now, coarse2fine
        logging.debug(f"coarse2fine!")
        zv, fine_zv_mask = self.coarse_to_fine(
            zv, 
            mask=mask,
            typical_filtering=True,
            _sampling_steps=2,
            return_mask=True
        )
        mask_z = torch.cat(
            [mask_z[:, :self.coarse.n_codebooks, :], fine_zv_mask[:, self.coarse.n_codebooks:, :]], 
            dim=1
        )

        z = zv

        if return_mask:
            return z, mask_z.cpu(),
        else:
            return z

    def visualize_codes(self, z: torch.Tensor):
        import matplotlib.pyplot as plt
        # make sure the figsize is square when imshow is called
        fig = plt.figure(figsize=(10, 7))
        # in subplots, plot z[0] and the mask
        # set title to "codes" and "mask"
        fig.add_subplot(2, 1, 1)
        plt.imshow(z[0].cpu().numpy(), aspect='auto', origin='lower', cmap="tab20")
        plt.title("codes")
        plt.ylabel("codebook index")
        # set the xticks to seconds


if __name__ == "__main__":
    import audiotools as at
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    torch.set_logging.debugoptions(threshold=10000)
    at.util.seed(42)

    interface = Interface(
        coarse_ckpt="./models/vampnet/coarse.pth", 
        coarse2fine_ckpt="./models/vampnet/c2f.pth", 
        codec_ckpt="./models/vampnet/codec.pth",
        device="cuda", 
        wavebeat_ckpt="./models/wavebeat.pth"
    )


    sig = at.AudioSignal('assets/example.wav')

    z = interface.encode(sig)


    mask = interface.build_mask(
        z=z,
        sig=sig,
        rand_mask_intensity=1.0,
        prefix_s=0.0,
        suffix_s=0.0,
        periodic_prompt=7,
        periodic_prompt2=7,
        periodic_prompt_width=1,
        onset_mask_width=5, 
        _dropout=0.0,
        upper_codebook_mask=3,
        upper_codebook_mask_2=None,
        ncc=0,
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
        breakpoint()

    mask = interface.decode(mask_z).cpu()

    sig = interface.decode(zv).cpu()


    logging.debug("done")

        