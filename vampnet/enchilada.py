import os
from pathlib import Path

import torch
from audiotools import AudioSignal

from .modules.transformer import VampNet
from lac.model.lac import LAC


class TheWholeEnchilada:
    def __init__(
        self,
        coarse_ckpt: str,
        coarse2fine_ckpt: str,
        codec_ckpt: str,
        device: str = "cpu",
    ):
        self.codec = LAC.load(Path(codec_ckpt))
        self.codec.eval()
        self.codec.to(device)

        self.coarse = VampNet.load(location=Path(coarse_ckpt), map_location="cpu")
        self.coarse.to(device)
        self.coarse.eval()

        self.coarse2fine = VampNet.load(
            location=Path(coarse2fine_ckpt), map_location="cpu"
        )
        # FIXME
        print(
            f"WARNING: PATCHING coarse2fine seq_len to 288, for backwards compatibility with a specific jazzpop model. it used to be {self.coarse2fine.seq_len}"
        )
        self.coarse2fine.seq_len = 288

        self.coarse2fine.to(device)
        self.coarse2fine.eval()

        self.device = device

    def seconds_to_tokens(self, seconds: float):
        return int(seconds * self.codec.sample_rate / self.codec.hop_length)

    def to(self, device):
        self.device = device
        self.coarse.to(device)
        self.coarse2fine.to(device)
        self.codec.to(device)
        return self

    def encode(self, signal: AudioSignal):
        with torch.inference_mode():
            # coarse z
            cz = self.codec.encode(signal.samples, signal.sample_rate)["codes"]

        return cz

    def vamp(
        self,
        signal,
        prefix_dur_s: float = 1.25,
        suffix_dur_s: float = 1.25,
        downsample_hint: bool = True,
        downsample_factor: int = 4,
        num_loops: int = 3,
        **kwargs,
    ):
        """
        Loop imputation of a signal.
        """
        signal.to(self.device).resample(self.codec.sample_rate).to_mono()

        z = self.encode(signal)

        cz = z[:, : self.coarse.n_codebooks, :].clone()
        original_cz = cz.clone()
        seq_len = original_cz.shape[-1]
        assert (
            seq_len == self.coarse.seq_len
        ), f"expected seq_len {self.coarse.seq_len}, got {seq_len} for token sequence length. Is your signal the same duration as the model was trained with? "

        vamp_hop_s = prefix_dur_s
        vamp_hop = self.seconds_to_tokens(vamp_hop_s)

        cmask = torch.ones_like(cz)

        if downsample_hint:
            # downsample by factor of 4
            for i in range(cmask.shape[-1]):
                if i % downsample_factor == 0:
                    cmask[:, :, i] = 0

        if prefix_dur_s > 0:
            prefix_len = self.seconds_to_tokens(prefix_dur_s)
            cmask[:, :, :prefix_len] = 0
            print(f"prefix_len: {prefix_len}")
        else:
            prefix_len = 0

        if suffix_dur_s > 0:
            suffix_len = self.seconds_to_tokens(suffix_dur_s)
            cmask[:, :, -suffix_len:] = 0
            print(f"suffix_len: {suffix_len}")
        else:
            suffix_len = 0

        prefix_z = cz[:, :, :prefix_len]

        coarse_vamp = [prefix_z.clone()]
        for i in range(num_loops):
            sampled_cz = self.coarse.sample(
                codec=self.codec,
                time_steps=seq_len,
                mask=cmask,
                start_tokens=cz,
                return_signal=False,
                **kwargs,
            )

            new_prefix = sampled_cz[:, :, prefix_len : prefix_len + vamp_hop]
            coarse_vamp.append(new_prefix.clone())

            # replace the prefix in cz with the new prefix
            # don't worry about a copy of the prefix still being
            # in the mask area, since that will be masked out
            cz[:, :, :vamp_hop] = new_prefix.clone()
            print("to append and to prefix")

        # we're done, so add the suffix
        coarse_vamp.append(sampled_cz[:, :, prefix_len + vamp_hop :])

        # concatenate the vamps
        coarse_vamp = torch.cat(coarse_vamp, dim=-1)

        # add a layer of
        fine_prefix = z[:, self.coarse.n_codebooks :, :prefix_len]
        fine_suffix = z[:, self.coarse.n_codebooks :, -suffix_len:]
        fine_vamp = torch.randint(
            0,
            self.coarse2fine.vocab_size,
            (
                coarse_vamp.shape[0],
                self.coarse2fine.n_predict_codebooks,
                coarse_vamp.shape[-1],
            ),
        ).to(self.device)
        fine_vamp[:, :, :prefix_len] = fine_prefix
        fine_vamp[:, :, -suffix_len:] = fine_suffix

        vamp_z = torch.cat([coarse_vamp, fine_vamp], dim=1)

        # now we sample from the coarse2fine model
        # to get the fine details
        start_pos = 0

        c2f_vamp = []
        while start_pos < vamp_z.shape[-1]:
            end_pos = min(start_pos + self.coarse2fine.seq_len, vamp_z.shape[-1])

            c2fz = vamp_z[:, :, start_pos:end_pos]
            self.coarse2fine: VampNet
            sampled_c2fz = self.coarse2fine.sample(
                codec=self.codec,
                start_tokens=c2fz,
                return_signal=False,
                mask=None,
            )
            c2f_vamp.append(sampled_c2fz)
            start_pos += self.coarse2fine.seq_len

        c2f_vamp = torch.cat(c2f_vamp, dim=-1)

        # make it a signal
        vamp_signal = self.coarse2fine.to_signal(c2f_vamp, self.codec)

        return {
            "full": vamp_signal,
            "coarse": self.coarse.to_signal(coarse_vamp, self.codec),
        }
