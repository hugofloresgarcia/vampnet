from audiotools import AudioSignal
import torch
import vampnet.mask as pmask
from vampnet.modules.transformer import VampNet
from dac.utils import load_model as load_dac
import math
import numpy as np
from tqdm import tqdm

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
    
    def vamp(self, 
        z, 
        mask, 
        return_mask=False, 
        gen_fn=None, 
        z_onsets=None,
        **kwargs
    ):
        # coarse z
        cz = z[:, : self.vampnet.n_codebooks, :].clone()
        mask = mask[:, : self.vampnet.n_codebooks, :]

        seq_len = cz.shape[-1]

        # we need to split the sequence into chunks by max seq length
        # we need to split so that the sequence length is less than the max_seq_len
        n_chunks = math.ceil(seq_len / self.vampnet.max_seq_len)
        chunk_len = math.ceil(seq_len / n_chunks)
        print(f"will process {n_chunks} chunks of length {chunk_len}")

        z_chunks = torch.split(cz, chunk_len, dim=-1)
        mask_chunks = torch.split(mask, chunk_len, dim=-1)

        gen_fn = gen_fn or self.vampnet.generate
        c_vamp_chunks = [
            gen_fn(
                codec=self.codec,
                time_steps=chunk.shape[-1],
                start_tokens=chunk,
                mask=mask_chunk,
                return_signal=False,
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
            periodic_prompt: int = 7,
            periodic_prompt_width: int = 1,
            onset_mask_width: int = 0, 
            dropout: float = 0.0,
            upper_codebook_mask: int = 3
    ):
        
        mask = pmask.linear_random(z, rand_mask_intensity)
        mask = pmask.mask_and(
            mask,
            pmask.inpaint(z, self.s2t(prefix_s), self.s2t(suffix_s)),
        )
        mask = pmask.mask_and(
            mask,
            pmask.periodic_mask(z, periodic_prompt, periodic_prompt_width, random_roll=True),
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

        # print(f"reloading to {data[model_choice]}")
        # self.reload(MODEL_CHOICES[data[model_choice]])

        prev_zvs = []
        for i in tqdm(range(feedback_steps), desc="feedback steps"):
            print(z.shape)
            mask = self.build_mask(
                z=z,
                **build_mask_kwargs
            )

            mask = mask.expand(batch_size, -1, -1)

            zv, mask_z = self.vamp(
                z,
                mask=mask,
                return_mask=True, 
                **vamp_kwargs
            )

            prev_zvs.append(zv)
            z = zv

        sig = self.to_signal(zv).cpu()
        print("done")

        sig = sig.normalize(loudness)

        if return_mask:
            return sig, mask.cpu()
        else:
            return sig.path_to_file

    def plot_sig_with_mask(self, sig, mask):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        sigout[0].specshow()
        plt.subplot(2, 1, 2)
        # plot the mask (which is a matrix)
        plt.imshow(mask[0].cpu().numpy(), aspect='auto', origin='lower', cmap='gray_r')
        plt.show()