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
from .signal import cut_to_hop_length, write

# from vampnet.dac.model.dac import DAC
# from vampnet.dac.model.dac import DAC
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

        # PATCH: the embedding layer has an unintitialized quantizer 
        self.coarse.embedding.quantizer = self.codec.quantizer
        self.c2f.embedding.quantizer = self.codec.quantizer

        self.device = device
        self.loudness = -24.0

        if compile:
            logging.debug(f"compiling models")
            self.coarse = torch.compile(self.coarse)
            if self.c2f is not None:
                self.c2f = torch.compile(self.c2f)
            self.codec = torch.compile(self.codec)


    @classmethod
    @torch.jit.ignore
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
    @torch.jit.ignore
    def available_models(cls):
        from . import list_finetuned
        return list_finetuned() + ["default"]


    @torch.jit.ignore
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

    @torch.jit.ignore
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
        return math.floor(seconds * self.codec.sample_rate / self.codec.hop_length)

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
    
    def _preprocess(self, signal: AudioSignal):
        signal = (
            signal.clone()
            .resample(self.codec.sample_rate)
            .to_mono()
            .normalize(self.loudness)
            .ensure_max_of_audio(1.0)
        )
        logging.debug(f"length before codec preproc: {signal.samples.shape}")
        signal.samples = self.codec.preprocess(signal.samples, signal.sample_rate)
        logging.debug(f"length after codec preproc: {signal.samples.shape}")
        return signal
    
    @torch.inference_mode()
    def encode(self, signal: AudioSignal):
        signal = signal.to(self.device)
        signal = self._preprocess(signal)
        z = self.codec.encode(signal.samples, signal.sample_rate)["codes"]
        return z

    @torch.inference_mode()
    def decode(self, z):
        """
        convert a sequence of latents to a signal. 
        """
        assert z.ndim == 3
        codec = self.codec

        # remove mask token
        z = z.masked_fill(z == self.coarse.mask_token, 0)
        signal = at.AudioSignal(
            codec.decode(
                codec.quantizer.from_latents(self.coarse.embedding.from_codes(z))[0]
            ),
            codec.sample_rate,
        )

        # find where the mask token is and replace it with silence in the audio
        for tstep in range(z.shape[-1]):
            if torch.all(z[:, :, tstep] == self.coarse.mask_token):
                sample_idx_0 = tstep * codec.hop_length
                sample_idx_1 = sample_idx_0 + codec.hop_length
                signal.samples[:, :, sample_idx_0:sample_idx_1] = 0.0

        return signal

    def set_chunk_size(self, chunk_size_s: float):
        self.coarse.chunk_size_s = chunk_size_s
        
    @torch.inference_mode()
    def coarse_to_fine(
        self, 
        z: Tensor,
        mask: Tensor = None,
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
                    start_tokens=chunk,
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
                    start_tokens=cz_masked_chunk,
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
        z: Tensor,
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
        codes: Tensor,
        mask: Tensor,
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
        zv, mask_z = self.coarse_vamp(
            z,
            mask=mask,
            return_mask=True, 
            **kwargs
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

    @torch.jit.ignore
    def visualize_codes(self, z: Tensor):
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
        zv = self.coarse.generate(
            start_tokens=z,
            mask=mask,
            temperature=temperature,
            typical_mass=typical_mass,
            typical_min_tokens=typical_min_tokens,
            seed=seed,
        )

        return zv

    @torch.inference_mode()
    def decode(self, z):
        return self.codec.decode(
            self.codec.quantizer.from_latents(self.coarse.embedding.from_codes(z))[0]
        )


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
    eiface = EmbeddedInterface(
        codec=interface.codec, 
        coarse=interface.coarse,
    )

    # handle roughly 10 seconds
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

        