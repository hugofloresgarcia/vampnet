import os
from pathlib import Path
import math

import torch
from audiotools import AudioSignal

from .modules.transformer import VampNet
from lac.model.lac import LAC


class Interface:
    def __init__(
        self,
        coarse_ckpt: str,
        coarse2fine_ckpt: str,
        codec_ckpt: str,
        device: str = "cpu",
        coarse_chunk_size_s: int =  5, 
        coarse2fine_chunk_size_s: int =  3,
    ):
        self.codec = LAC.load(Path(codec_ckpt))
        self.codec.eval()
        self.codec.to(device)

        self.coarse = VampNet.load(location=Path(coarse_ckpt), map_location="cpu")
        self.coarse.to(device)
        self.coarse.eval()
        self.coarse.chunk_size_s = coarse_chunk_size_s

        self.c2f = VampNet.load(
            location=Path(coarse2fine_ckpt), map_location="cpu"
        )
        self.c2f.to(device)
        self.c2f.eval()
        self.c2f.chunk_size_s = coarse2fine_chunk_size_s

        self.device = device

    def s2t(self, seconds: float):
        """seconds to tokens"""
        return int(seconds * self.codec.sample_rate / self.codec.hop_length)

    def to(self, device):
        self.device = device
        self.coarse.to(device)
        self.c2f.to(device)
        self.codec.to(device)
        return self

    def to_signal(self, z: torch.Tensor):
        return self.coarse.to_signal(z, self.codec)
    
    @torch.inference_mode()
    def encode(self, signal: AudioSignal):
        signal = signal.clone().to(self.device).resample(self.codec.sample_rate).to_mono()
        z = self.codec.encode(signal.samples, signal.sample_rate)["codes"]
        return z

    def coarse_to_fine(
        self, 
        coarse_z: torch.Tensor,
        **kwargs
    ):
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
            chunk = self.c2f.sample(
                codec=self.codec,
                time_steps=chunk_len,
                start_tokens=chunk,
                return_signal=False,
            )
            fine_z.append(chunk)

        fine_z = torch.cat(fine_z, dim=-1)
        return fine_z[:, :, :length].clone()
    
    def coarse_vamp(
        self, 
        signal, 
        prefix_dur_s: float = 1.25, 
        suffix_dur_s: float = 1.25,
        num_loops: int = 3,
        mode="impute",
        downsample_factor: int = None,
        debug=False,
        **kwargs
    ):
        z = self.encode(signal)

        assert signal.duration == self.coarse.chunk_size_s, "signal duration must match coarse chunk size for now"

        # coarse z
        cz = z[:, : self.coarse.n_codebooks, :].clone()
        c_seq_len = cz.shape[-1]
        n_prefix = self.s2t(prefix_dur_s)
        n_suffix = self.s2t(suffix_dur_s)
        
        # we'll keep the final codes sequence here
        c_vamp = {
            'prefix': [cz[:, :, :n_prefix].clone()],
            'suffix': [cz[:, :, c_seq_len-n_suffix:].clone()]
        }

        _cz = cz.clone()
        for _ in range(num_loops):
            # add noise
            cz_masked, cz_mask = self.coarse.add_noise(
                _cz, r=0.0,
                n_prefix=n_prefix,
                n_suffix=n_suffix, 
                downsample_factor=downsample_factor
            )
            if debug:
                print("tokens to infer")
                self.to_signal(cz_masked).cpu().widget()

            # sample!
            cz_sampled = self.coarse.sample(
                codec=self.codec,
                time_steps=self.s2t(self.coarse.chunk_size_s),
                start_tokens=_cz,
                mask=cz_mask, 
                return_signal=False,
                **kwargs
            )

            if debug:
                print("tokens sampled")
                self.to_signal(cz_sampled).cpu().widget()
            
            cz_imputed = cz_sampled[:, :, n_prefix:c_seq_len-n_suffix].clone()
            
            if mode == "impute":
                 # split the imputed codes into two halves
                cz_imputed_a = cz_imputed[:, :, : cz_imputed.shape[-1] // 2].clone()
                cz_imputed_b = cz_imputed[:, :, cz_imputed.shape[-1] // 2 :].clone()
            elif mode == "continue":
                cz_imputed_a = cz_imputed[:, :, : cz_imputed.shape[-1]].clone()
                cz_imputed_b = _cz[:, :, :0].clone() # empty 
            elif mode == "reverse-continue":
                cz_imputed_a = _cz[:, :, :0].clone() # empty
                cz_imputed_b = cz_imputed[:, :, : cz_imputed.shape[-1]].clone()
            else:
                raise ValueError(f"mode {mode} not supported")

            if debug:
                # add to our c_vamp
                if cz_imputed_a.shape[-1] > 0:
                    print("new_prefix added")
                    self.to_signal(cz_imputed_a).cpu().widget()
                if cz_imputed_b.shape[-1] >  0:
                    print("new_suffix added")   
                    self.to_signal(cz_imputed_b).cpu().widget()

            c_vamp['prefix'].append(cz_imputed_a.clone())
            c_vamp['suffix'].insert(0, cz_imputed_b.clone())

            n_to_insert = c_seq_len - (cz_imputed_a.shape[-1] + cz_imputed_b.shape[-1])
            to_insert = torch.zeros(cz_imputed_a.shape[0], cz_imputed_a.shape[1], n_to_insert).long().to(self.device)
            _cz = torch.cat([cz_imputed_a, to_insert, cz_imputed_b], dim=-1)

            if debug:
                print("tokens to infer next round (area to insert in the middle)")
                self.to_signal(_cz).cpu().widget()




        prefix_codes = torch.cat(c_vamp['prefix'], dim=-1)
        suffix_codes = torch.cat(c_vamp['suffix'], dim=-1)
        c_vamp = torch.cat([prefix_codes, suffix_codes], dim=-1)
        return c_vamp


    def coarse_vamp_v2(
        self, 
        signal, 
        prefix_dur_s: float = 1.25, 
        suffix_dur_s: float = 1.25,
        num_loops: int = 3,
        downsample_factor: int = None,
        debug=False,
        **kwargs
    ):
        z = self.encode(signal)

        assert signal.duration == self.coarse.chunk_size_s, "signal duration must match coarse chunk size for now"

        # coarse z
        cz = z[:, : self.coarse.n_codebooks, :].clone()
        c_seq_len = cz.shape[-1]
        n_prefix = self.s2t(prefix_dur_s)
        n_suffix = self.s2t(suffix_dur_s)

        assert n_prefix + n_suffix < c_seq_len, "prefix and suffix must be smaller than the chunk size"
        
        # we'll keep the final codes sequence here
        c_vamp = {
            'prefix': [cz[:, :, :n_prefix].clone()],
            'suffix': [cz[:, :, c_seq_len-n_suffix:].clone()]
        }

        _cz = cz.clone()
        cz_mask = None
        for _ in range(num_loops):
            # add noise
            cz_masked, cz_mask = self.coarse.add_noise(
                _cz, r=0.0,
                n_prefix=n_prefix,
                n_suffix=n_suffix, 
                downsample_factor=downsample_factor,
                mask=cz_mask
            )
            if debug:
                print("tokens to infer")
                self.to_signal(cz_masked).cpu().widget()

            # sample!
            if debug:
                print(f"mask: {cz_mask[:,0,:]}")
                print(f"z: {_cz[:,0,:]}")
            cz_sampled = self.coarse.sample(
                codec=self.codec,
                time_steps=self.s2t(self.coarse.chunk_size_s),
                start_tokens=_cz,
                mask=cz_mask, 
                return_signal=False,
                **kwargs
            )

            if debug:
                print("tokens sampled")
                self.to_signal(cz_sampled).cpu().widget()
            
            # the z that was generated
            cz_generated = cz_sampled[:, :, n_prefix:c_seq_len-n_suffix].clone()
            n_generated = cz_generated.shape[-1]

            # create the new prefix and suffix
            # we'll make sure that the number of prefix and suffix
            # tokens is the same as the original
            # but we do want to advance the sequence as much as we can
            if n_prefix > 0 and n_suffix > 0:
                # we have both prefix and suffix, so we'll split the generated
                # codes in two halves
                prefix_start_idx = n_generated // 2
                prefix_stop_idx = prefix_start_idx + n_prefix
                assert prefix_start_idx >= 0, "internal error"

                suffix_start_idx = n_prefix + n_generated // 2
                suffix_stop_idx = suffix_start_idx + n_suffix
                assert suffix_stop_idx <= cz_sampled.shape[-1], "internal error"

                cz_new_prefix = cz_sampled[:, :, prefix_start_idx:prefix_stop_idx].clone()
                cz_new_suffix = cz_sampled[:, :, suffix_start_idx:suffix_stop_idx].clone()

                c_vamp['prefix'].append(cz_generated[:,:,:n_generated//2])
                c_vamp['suffix'].insert(0, cz_generated[:,:,n_generated//2:])

            elif n_prefix > 0:
                # we only have a prefix
                prefix_start_idx = n_generated
                prefix_stop_idx = prefix_start_idx + n_prefix

                cz_new_prefix = cz_sampled[:, :, prefix_start_idx:prefix_stop_idx].clone()
                cz_new_suffix = _cz[:, :, :0].clone()
                

                c_vamp['prefix'].append(cz_generated)

            elif n_suffix > 0:
                # we only have a suffix, so everything starting at 0 is generated
                suffix_stop_idx = max(n_generated, n_suffix)
                suffix_start_idx = suffix_stop_idx - n_suffix

                cz_new_prefix = _cz[:, :, :0].clone()
                cz_new_suffix = cz_sampled[:, :, suffix_start_idx:suffix_stop_idx].clone()

                c_vamp['suffix'].insert(0, cz_generated)

            
            n_to_insert = c_seq_len - (cz_new_prefix.shape[-1] + cz_new_suffix.shape[-1])
            to_insert = torch.zeros(cz_new_prefix.shape[0], cz_new_prefix.shape[1], n_to_insert).long().to(self.device)
            _cz = torch.cat([cz_new_prefix, to_insert, cz_new_suffix], dim=-1)

            to_insert_mask = torch.zeros_like(_cz).long().to(self.device)
            to_insert_mask[:, :, cz_new_prefix.shape[-1]:cz_new_prefix.shape[-1]+n_to_insert] = 1
            cz_mask = (cz_mask + to_insert_mask).bool().long()


            if debug:
                print("tokens to infer next round (area to insert in the middle)")
                self.to_signal(_cz).cpu().widget()


        prefix_codes = torch.cat(c_vamp['prefix'], dim=-1)
        suffix_codes = torch.cat(c_vamp['suffix'], dim=-1)
        c_vamp = torch.cat([prefix_codes, suffix_codes], dim=-1)
        return c_vamp












        


