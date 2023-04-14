import os
from pathlib import Path
import math

import torch
import numpy as np
from audiotools import AudioSignal
import tqdm

from .modules.transformer import VampNet
from .beats import WaveBeat
from lac.model.lac import LAC


def signal_concat(
    audio_signals: list,
):
    audio_data = torch.cat([x.audio_data for x in audio_signals], dim=-1)

    return AudioSignal(audio_data, sample_rate=audio_signals[0].sample_rate)


class Interface:
    def __init__(
        self,
        coarse_ckpt: str = None,
        coarse2fine_ckpt: str = None,
        codec_ckpt: str = None,
        wavebeat_ckpt: str = None,
        device: str = "cpu",
        coarse_chunk_size_s: int =  5, 
        coarse2fine_chunk_size_s: int =  3,
    ):
        assert codec_ckpt is not None, "must provide a codec checkpoint"
        self.codec = LAC.load(Path(codec_ckpt))
        self.codec.eval()
        self.codec.to(device)

        assert coarse_ckpt is not None, "must provide a coarse checkpoint"
        self.coarse = VampNet.load(location=Path(coarse_ckpt), map_location="cpu")
        self.coarse.to(device)
        self.coarse.eval()
        self.coarse.chunk_size_s = self.s2t2s(coarse_chunk_size_s)

        if coarse2fine_ckpt is not None:
            self.c2f = VampNet.load(
                location=Path(coarse2fine_ckpt), map_location="cpu"
            )
            self.c2f.to(device)
            self.c2f.eval()
            self.c2f.chunk_size_s = self.s2t2s(coarse2fine_chunk_size_s)
        else:
            self.c2f = None

        if wavebeat_ckpt is not None:
            print(f"loading wavebeat from {wavebeat_ckpt}")
            self.beat_tracker = WaveBeat(wavebeat_ckpt)
            self.beat_tracker.model.to(device)
        else:
            self.beat_tracker = None

        self.device = device

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
    
    def autoencode(self, signal: AudioSignal):
        z = self.encode(signal)
        return self.to_signal(z)
    
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
            dropout: float = 0.3,
            invert: bool = True,
    ):
        """make a beat synced mask. that is, make a mask that 
        places 1s at and around the beat, and 0s everywhere else. 
        """
        assert hasattr(self, "beat_tracker"), "No beat tracker loaded"

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
                _m = torch.nn.functional.dropout(_m, p=dropout)
                
                mask[_slice[0]:_slice[1]] = _m

        if mask_downbeats:
            for downbeat_idx in downbeats_z:
                _slice = int(downbeat_idx - mask_b4), int(downbeat_idx + mask_after)
                num_steps = mask[_slice[0]:_slice[1]].shape[0]
                _m = torch.ones(num_steps, device=self.device)
                _m = torch.nn.functional.dropout(_m, p=dropout)
                
                mask[_slice[0]:_slice[1]] = _m
        
        mask = mask.clamp(0, 1)
        if invert:
            mask = 1 - mask
        
        return mask[None, None, :].bool().long()
        
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
            chunk = self.c2f.sample(
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
        prefix_dur_s: float = 0.0, 
        suffix_dur_s: float = 0.0,
        num_vamps: int = 1,
        downsample_factor: int = None,
        intensity: float = 1.0, 
        debug=False,
        swap_prefix_suffix=False, 
        ext_mask=None,
        n_conditioning_codebooks=None,
        verbose=False,
        return_mask=False,
        **kwargs
    ):
        z = self.encode(signal)

        # coarse z
        cz = z[:, : self.coarse.n_codebooks, :].clone()
        c_seq_len = cz.shape[-1]
        n_prefix = self.s2t(prefix_dur_s)
        n_suffix = self.s2t(suffix_dur_s)

        assert cz.shape[-1] <= self.s2t(self.coarse.chunk_size_s), f"the sequence of tokens provided must match the one specified in the coarse chunk size, but got {cz.shape[-1]} and {self.s2t(self.coarse.chunk_size_s)}"
        assert n_prefix + n_suffix < c_seq_len, "prefix and suffix must be smaller than the chunk size"

        if swap_prefix_suffix:
            # swap the prefix and suffix regions in c_z
            assert n_prefix == n_suffix, "prefix and suffix must be the same size for now"
            cz[:, :, :n_prefix], cz[:, :, c_seq_len-n_suffix:] = cz[:, :, c_seq_len-n_suffix:], cz[:, :, :n_prefix].clone()
        
        # we'll keep the final codes sequence here
        c_vamp = {
            'prefix': [cz[:, :, :n_prefix].clone()],
            'suffix': [cz[:, :, c_seq_len-n_suffix:].clone()]
        }

        _cz = cz.clone()
        cz_mask = None
        range_fn = tqdm.trange if verbose else range
        for _ in range_fn(num_vamps):
            # add noise
            cz_masked, cz_mask = self.coarse.add_noise(
                _cz, r=1.0-intensity,
                n_prefix=n_prefix,
                n_suffix=n_suffix, 
                downsample_factor=downsample_factor,
                mask=cz_mask, 
                ext_mask=ext_mask, 
                n_conditioning_codebooks=n_conditioning_codebooks
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
                time_steps=_cz.shape[-1],
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

            else:
                # we have no prefix or suffix, so we'll just use the generated
                # codes as the new prefix and suffix
                cz_new_prefix = cz_generated.clone()
                cz_new_suffix = _cz[:, :, :0].clone()

                c_vamp['prefix'].append(cz_generated)

            
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

        if return_mask:
            return c_vamp, cz_masked
        return c_vamp

    # create a variation of an audio signal
    def variation(
        self, 
        signal: AudioSignal, 
        verbose: bool = False,
        beat_mask: bool = False,
        beat_mask_kwargs: dict = {}, 
        **kwargs
    ):
        signal = signal.clone()

        # autoencode first, so the samples get rounded up to the nearest tokens
        signal = self.autoencode(signal).cpu()

        # pad the signal to the nearest chunk size
        req_len = (
            math.ceil(signal.duration / self.coarse.chunk_size_s) 
            * self.coarse.chunk_size_s
        )
        # eventually we DO want overlap, but we want overlap-replace not
        # overlap-add
        overlap_hop_ratio = 1.0
        hop_duration = self.coarse.chunk_size_s * overlap_hop_ratio
        original_length = signal.length

        signal.zero_pad_to(req_len)

        # window the signal
        signal = signal.collect_windows(
            window_duration=self.coarse.chunk_size_s,
            hop_duration=hop_duration,
        )

        # output = []
        range_fn = range if not verbose else tqdm.trange
        for i in range_fn(signal.batch_size):
            sig = AudioSignal(
                signal.samples[i,...], signal.sample_rate
            )
            sig.to(self.device)

            if beat_mask:
                ext_mask = self.make_beat_mask(sig, **beat_mask_kwargs)
            else:
                ext_mask = None
            
            out_z = self.coarse_vamp_v2(
                sig, 
                num_vamps=1, 
                swap_prefix_suffix=False, 
                ext_mask=ext_mask,
                verbose=verbose,
                **kwargs
            )
            if self.c2f is not None:
                out_z = self.coarse_to_fine(out_z)
            out_sig = self.to_signal(out_z).cpu()

            signal.samples[i] = out_sig.samples

        output = signal.overlap_and_add(hop_duration)

        output.truncate_samples(original_length)
        return output

    # create a loop of a single region with variations
    # TODO: this would work nicer if we could trim at the beat
    # otherwise the model has to awkwardly fill up space that won't match
    # the beat unless the signal is exactly the right length
    def loop(
        self, 
        signal: AudioSignal, 
        prefix_dur_s: float = 0.0,
        suffix_dur_s: float = 0.0,
        num_loops: int = 4, 
        # overlap_hop_ratio: float = 1.0, # TODO: should this be fixed to 1.0?  or should we overlap and replace instead of overlap add
        verbose: bool = False,
        return_mask: bool = False,
        **kwargs, 
    ):
        assert prefix_dur_s >= 0.0, "prefix duration must be >= 0"
        assert suffix_dur_s >= 0.0, "suffix duration must be >= 0"
        signal = self.preprocess(signal)

        suffix_len_samples = int(suffix_dur_s * signal.sample_rate)
        prefix_len_tokens = self.s2t(prefix_dur_s)
        suffix_len_tokens = self.s2t(suffix_dur_s)

        loops = [
            # add everything but the suffix a the beggining
            self.encode(signal.clone().trim(before=0, after=suffix_len_samples))
        ]
        range_fn = range if not verbose else tqdm.trange
        for i in range_fn(num_loops):
            is_flipped = i % 2 == 0
            vamped = self.coarse_vamp_v2(
                        signal, 
                        prefix_dur_s=prefix_dur_s,
                        suffix_dur_s=suffix_dur_s,
                        swap_prefix_suffix=is_flipped,
                        return_mask=return_mask,
                        **kwargs
                )
            if return_mask:
                vamped, mask = vamped
            
            # if we're flipped, we trim the prefix off of the end
            # otherwise we trim the suffix off of the end
            trim_len = prefix_len_tokens if is_flipped else suffix_len_tokens
            vamped = vamped[:, :, :vamped.shape[-1]-trim_len]

            loops.append(vamped)

        if is_flipped:
            loops.append(
                # add everything but the prefix at the end
                self.encode(signal.clone())
            )

        if self.c2f is not None:
            loops = [self.coarse_to_fine(l) for l in loops]

        loops = [self.to_signal(l) for l in loops]

        if return_mask:
            return signal_concat(loops), self.to_signal(mask)
        
        return signal_concat(loops)

