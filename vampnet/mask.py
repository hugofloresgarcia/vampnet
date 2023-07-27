from typing import Optional

import torch
from audiotools import AudioSignal

from .util import scalar_to_batch_tensor

def _gamma(r):
    return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

def _invgamma(y):
    if not torch.is_tensor(y):
        y = torch.tensor(y)[None]
    return 2 * y.acos() / torch.pi

def full_mask(x: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    return torch.ones_like(x).long()

def empty_mask(x: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    return torch.zeros_like(x).long()

def apply_mask(
        x: torch.Tensor, 
        mask: torch.Tensor, 
        mask_token: int
    ):
    assert mask.ndim == 3, "mask must be (batch, n_codebooks, seq), but got {mask.ndim}"
    assert mask.shape == x.shape, f"mask must be same shape as x, but got {mask.shape} and {x.shape}" 
    assert mask.dtype == torch.long, "mask must be long dtype, but got {mask.dtype}"
    assert ~torch.any(mask > 1), "mask must be binary"
    assert ~torch.any(mask < 0), "mask must be binary"

    fill_x = torch.full_like(x, mask_token)
    x = x * (1 - mask) + fill_x * mask

    return x, mask

def random(
    x: torch.Tensor,
    r: torch.Tensor
):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)

    r = _gamma(r)[:, None, None]
    probs = torch.ones_like(x) * r

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask

def linear_random(
    x: torch.Tensor,
    r: torch.Tensor,
):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device).float()

    probs = torch.ones_like(x).to(x.device).float()
    # expand to batch and codebook dims
    probs = probs.expand(x.shape[0], x.shape[1], -1)
    probs = probs * r

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask

def inpaint(x: torch.Tensor, 
    n_prefix,
    n_suffix,
):
    assert n_prefix is not None
    assert n_suffix is not None
    
    mask = full_mask(x)

    # if we have a prefix or suffix, set their mask prob to 0
    if n_prefix > 0:
        if not isinstance(n_prefix, torch.Tensor):
            n_prefix = scalar_to_batch_tensor(n_prefix, x.shape[0]).to(x.device) 
        for i, n in enumerate(n_prefix):
            if n > 0:
                mask[i, :, :n] = 0.0
    if n_suffix > 0:
        if not isinstance(n_suffix, torch.Tensor):
            n_suffix = scalar_to_batch_tensor(n_suffix, x.shape[0]).to(x.device)
        for i, n in enumerate(n_suffix):
            if n > 0:
                mask[i, :, -n:] = 0.0

    
    return mask

def periodic_mask(x: torch.Tensor, 
                period: int, width: int = 1, 
                random_roll=False,
    ):
    mask = full_mask(x)
    if period == 0:
        return mask

    if not isinstance(period, torch.Tensor):
        period = scalar_to_batch_tensor(period, x.shape[0])
    for i, factor in enumerate(period):
        if factor == 0:
            continue
        for j in range(mask.shape[-1]):
            if j % factor == 0:
                # figure out how wide the mask should be
                j_start = max(0, j - width // 2  )
                j_end = min(mask.shape[-1] - 1, j + width // 2 ) + 1 
                # flip a coin for each position in the mask
                j_mask = torch.bernoulli(torch.ones(j_end - j_start))
                assert torch.all(j_mask == 1)
                j_fill = torch.ones_like(j_mask) * (1 - j_mask)
                assert torch.all(j_fill == 0)
                # fill
                mask[i, :, j_start:j_end] = j_fill
    if random_roll:
        # add a random offset to the mask
        offset = torch.randint(0, period[0], (1,))
        mask = torch.roll(mask, offset.item(), dims=-1)

    return mask

def codebook_unmask(
    mask: torch.Tensor, 
    n_conditioning_codebooks: int
):
    if n_conditioning_codebooks == None:
        return mask
    # if we have any conditioning codebooks, set their mask  to 0
    mask = mask.clone()
    mask[:, :n_conditioning_codebooks, :] = 0
    return mask

def mask_and(
    mask1: torch.Tensor, 
    mask2: torch.Tensor
):
    assert mask1.shape == mask2.shape, "masks must be same shape"
    return torch.min(mask1, mask2)

def dropout(
    mask: torch.Tensor,
    p: float,
):
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert mask.max() <= 1, "mask must be binary"
    assert mask.min() >= 0, "mask must be binary"
    mask = (~mask.bool()).float()
    mask = torch.bernoulli(mask * (1 - p))
    mask = ~mask.round().bool()
    return mask.long()

def mask_or(
    mask1: torch.Tensor, 
    mask2: torch.Tensor
):
    assert mask1.shape == mask2.shape, f"masks must be same shape, but got {mask1.shape} and {mask2.shape}"
    assert mask1.max() <= 1, "mask1 must be binary"
    assert mask2.max() <= 1, "mask2 must be binary"
    assert mask1.min() >= 0, "mask1 must be binary"
    assert mask2.min() >= 0, "mask2 must be binary"
    return (mask1 + mask2).clamp(0, 1)

def time_stretch_mask(
    x: torch.Tensor, 
    stretch_factor: int,
):
    assert stretch_factor >= 1, "stretch factor must be >= 1"
    c_seq_len = x.shape[-1]
    x = x.repeat_interleave(stretch_factor, dim=-1)

    # trim cz to the original length
    x = x[:, :, :c_seq_len]

    mask = periodic_mask(x, stretch_factor, width=1)
    return mask

def onset_mask(
    sig: AudioSignal, 
    z: torch.Tensor,
    interface,
    width: int = 1
):
    import librosa
    import madmom
    from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
    import tempfile
    import numpy as np 

    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        sig = sig.clone()
        sig.write(f.name)

        proc = RNNOnsetProcessor(online=False)
        onsetproc = OnsetPeakPickingProcessor(threshold=0.3,
                                              fps=sig.sample_rate/interface.codec.hop_length)
        
        act = proc(f.name)
        onset_times = onsetproc(act)

        # convert to indices for z array
        onset_indices = librosa.time_to_frames(onset_times, sr=sig.sample_rate, hop_length=interface.codec.hop_length)

        if onset_indices.shape[0] == 0:
            mask = empty_mask(z)   
            print(f"no onsets found, returning empty mask")
        else: 
            torch.set_printoptions(threshold=1000)
            print("onset indices: ", onset_indices)
            print("onset times: ", onset_times)

            # create a mask, set onset 
            mask = torch.ones_like(z)
            n_timesteps = z.shape[-1]

            for onset_index in onset_indices:
                onset_index = min(onset_index, n_timesteps - 1)
                onset_index = max(onset_index, 0)
                mask[:, :, onset_index - width:onset_index + width] = 0.0

            print(mask)
    
    return mask



if __name__ == "__main__":
    pass
