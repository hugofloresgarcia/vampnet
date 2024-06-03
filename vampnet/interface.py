from audiotools import AudioSignal
import torch
import vampnet.mask as pmask
from vampnet.model.transformer import VampNet
import math
import numpy as np
from tqdm import tqdm
from dac.model.dac import DAC
import audiotools as at
from typing import Optional

import vampnet

class Interface:

    def __init__(self, 
        codec: DAC, 
        model: VampNet, 
        device=vampnet.DEVICE
    ):
        self.codec = codec
        self.model = model

        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        self.codec.to(device)
        self.model.to(device)
        return self

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
        self.model.to(device)
        self.codec.to(device)
        return self

    @torch.no_grad()
    def decode(self, z,  silence_mask=True):
        """
        convert a sequence of latents to a signal.
        """
        assert z.ndim == 3
        codec = self.codec
        _z = codec.quantizer.from_latents(self.model.embedding.from_codes(z, codec))[0]

        signal = at.AudioSignal(
            codec.decode(_z), 
            codec.sample_rate,
        )

        if silence_mask:
            # find where the mask token is and replace it with silence in the audio
            for tstep in range(z.shape[-1]):
                if torch.any(z[:, :, tstep] == self.model.special_tokens["MASK"]):
                    sample_idx_0 = tstep * codec.hop_length
                    sample_idx_1 = sample_idx_0 + codec.hop_length
                    signal.samples[:, :, sample_idx_0:sample_idx_1] = 0.0

        return signal

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
    

    def vamp(self, 
        z: torch.Tensor, 
        mask: torch.Tensor, 
        return_mask:bool=False, 
        gen_fn: callable=None, 
        **kwargs
    ):
        """
        vamp on a sequence of codes z, given a mask. 

        Args:
            z (torch.Tensor): a sequence of codes. shape (batch_size, n_codebooks, seq_len)
            mask (torch.Tensor): a mask. shape (batch_size, n_codebooks, seq_len)
            return_mask (bool, optional): return the mask. Defaults to False.
            gen_fn (callable, optional): used for debugging only. a function to generate the codes.  Defaults to None.
        Returns:
            torch.Tensor: a vamped of codes. shape (batch_size, n_codebooks, seq_len)
        """
        # coarse z
        cz = z[:, : self.model.n_codebooks, :].clone()
        mask = mask[:, : self.model.n_codebooks, :]

        seq_len = cz.shape[-1]

        # we need to split the sequence into chunks by max seq length
        # we need to split so that the sequence length is less than the max_seq_len
        n_chunks = math.ceil(seq_len / self.model.max_seq_len)
        chunk_len = math.ceil(seq_len / n_chunks)
        print(f"will process {n_chunks} chunks of length {chunk_len}")

        z_chunks = torch.split(cz, chunk_len, dim=-1)
        mask_chunks = torch.split(mask, chunk_len, dim=-1)

        gen_fn = gen_fn or self.model.generate
        c_vamp_chunks = [
            gen_fn(
                codec=self.codec,
                time_steps=chunk.shape[-1],
                start_tokens=chunk,
                mask=mask_chunk,
                **kwargs,
            )
            for chunk, mask_chunk in tqdm(zip(z_chunks, mask_chunks), desc="vamping chunks")
        ]

        # concatenate the chunks
        c_vamp = torch.cat(c_vamp_chunks, dim=-1)

        if return_mask:
            return c_vamp, mask
        else:
            return c_vamp


    def build_mask(self, 
            z: torch.Tensor,
            rand_mask_intensity: float = 1.0,
            prefix_s: float = 0.0,
            suffix_s: float = 0.0,
            periodic_prompt: Optional[int] = 7,
            periodic_prompt_width: int = 1,
            onset_mask_width: int = 0, 
            upper_codebook_mask: Optional[int] = None,
            dropout: float = 0.0,
    ):
        """
        Build a mask for the a sequence of codes z. 

        Args:
            z (torch.Tensor): a sequence of codes. shape (batch_size, n_codebooks, seq_len)
            rand_mask_intensity (float, optional): intensity of the random mask. Defaults to 1.0. 
            prefix_s (float, optional): length of the prefix mask in seconds. Defaults to 0.0.
            suffix_s (float, optional): length of the suffix mask in seconds. Defaults to 0.0.
            periodic_prompt (Optional[int], optional): periodic prompt. for a period of `p`, this will mask everything but every `p`th token. Defaults to 7.
            periodic_prompt_width (int, optional): periodic prompt width. values larger than 1 will create a wider periodic prompt. Defaults to 1.
            onset_mask_width (int, optional): DEPRECATED. Defaults to 0.
            dropout (float, optional): perform any dropout on the final mask. Defaults to 0.0.
            upper_codebook_mask (Optional[int], optional): if a number `n` is given, then any tokens above codebook level `n` will be masked. Defaults to None.
        
        Returns: a mask the same size as z shape (batch_size, n_codebooks, seq_len)
        """
        if upper_codebook_mask is None:
            upper_codebook_mask = self.model.n_codebooks
        if periodic_prompt is None:
            periodic_prompt = self.model.n_codebooks
        
        mask = pmask.linear_random(z, rand_mask_intensity)
        mask = pmask.mask_and(
            mask,
            pmask.inpaint(z, self.s2t(prefix_s), self.s2t(suffix_s)),
        )
        mask = pmask.mask_and(
            mask,
            pmask.periodic(z, periodic_prompt, periodic_prompt_width, random_roll=True),
        )
        if onset_mask_width > 0:
            raise NotImplementedError("onset mask not implemented. currently depends on sig which breaks the code structure.")
            mask = pmask.mask_and(
                mask,
                pmask.onset_mask(sig, z, onset_mask_width, self),
            )

        mask = pmask.dropout(mask, dropout)
        mask = pmask.codebook_mask(mask, int(upper_codebook_mask))
        return mask


    def ez_vamp(
        self, 
        sig: AudioSignal,
        batch_size: int = 4,
        feedback_steps: int = 1,
        return_mask: bool = False,
        build_mask_kwargs: dict = None,
        vamp_kwargs: dict = None,
    ):
        build_mask_kwargs = build_mask_kwargs or {}
        vamp_kwargs = vamp_kwargs or {}

        sig = self.preprocess(sig)
        loudness = sig.loudness()

        z = self.encode(sig)

        # expand z to batch size
        z = z.expand(batch_size, -1, -1)

        prev_zvs = []
        for i in tqdm(range(feedback_steps), desc="feedback steps"):
            print(z.shape)
            mask = self.build_mask(
                z=z,
                **build_mask_kwargs
            )

            mask = mask.expand(batch_size, -1, -1)

            vamp_kwargs.pop("mask", None)
            vamp_kwargs.pop('return_mask', None)
            zv, mask_z = self.vamp(
                z,
                mask=mask,
                return_mask=True, 
                **vamp_kwargs
            )

            prev_zvs.append(zv)
            z = zv

        sig = self.decode(zv).cpu()
        print("done")

        sig = sig.normalize(loudness)

        if return_mask:
            return sig, mask.cpu()
        else:
            return sig


    def plot_sig_with_mask(self, sig, mask):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        sig[0].specshow()
        plt.subplot(2, 1, 2)
        # plot the mask (which is a matrix)
        plt.imshow(mask[0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray_r')
        plt.show()