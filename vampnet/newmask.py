from typing import Optional

import torch

from .util import scalar_to_batch_tensor

def _gamma(r):
    return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

def _invgamma(y):
    if not torch.is_tensor(y):
        y = torch.tensor(y)[None]
    return 2 * y.acos() / torch.pi

def full_mask(x: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    return torch.ones_like(x).int()

def empty_mask(x: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    return torch.zeros_like(x).int()

def apply_mask(
        x: torch.Tensor, 
        mask: torch.Tensor, 
        mask_token: int
    ):
    assert mask.ndim == 3, f"mask must be (batch, n_codebooks, seq), but got {mask.ndim}"
    assert mask.shape == x.shape, f"mask must be same shape as x, but got {mask.shape} and {x.shape}" 
    assert mask.dtype == torch.int, f"mask must be int dtype, but got {mask.dtype}"
    assert ~torch.any(mask > 1), "mask must be binary"
    assert ~torch.any(mask < 0), "mask must be binary"
    mask = mask.int()

    fill_x = torch.full_like(x, mask_token)
    x = x * (1 - mask) + fill_x * mask

    return x

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
    mask = mask.round().int()

    return mask, torch.zeros_like(mask).bool()

def random_along_time(x: torch.Tensor, r: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, channel, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)
    
    x = x[:, 0, :]
    r = _gamma(r)[:, None]
    probs = torch.ones_like(x) * r

    mask = torch.bernoulli(probs)
    mask = mask.round().int()

    return mask
    

def stemgen_random(x: torch.Tensor, r: torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)

    # Assuming x is your input tensor and r is the probability for the Bernoulli distribution
    nb, nc, nt = x.shape

    # Randomly sample a codebook level to infer for each item in the batch
    c = torch.randint(0, nc, (nb,)).to(x.device)

    # Create a mask tensor of the same shape as x, initially filled with ones
    mask = torch.ones_like(x).long().to(x.device)
    ignore_indices_mask = torch.zeros_like(x).long().to(x.device)

    # Iterate over each item in the batch
    for i in range(nb):
        # Create the Bernoulli mask for the sampled level
        level_mask = torch.bernoulli(torch.ones(nt).to(x.device) * r[i]).long()

        # Apply the mask to the sampled level
        mask[i, c[i]] = level_mask

        # All levels below the sampled level are unmasked (zeros)
        mask[i, :c[i]] = 0
        ignore_indices_mask[i, :c[i]] = 1

        # All levels above the sampled level are masked (ones)
        mask[i, c[i]+1:] = 1
        ignore_indices_mask[i, c[i]+1:] = 1

    # save a debug mask to np txt
    # import numpy as np
    # np.savetxt("mask.txt", mask[0].cpu().numpy(), fmt="%d")
    # np.savetxt("ignore_indices_mask.txt", ignore_indices_mask[0].cpu().numpy(), fmt="%d")

    return mask.int(), ignore_indices_mask.bool()


def hugo_random(x: torch.Tensor, r:torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device).float()
    
    r = _gamma(r)[:, None, None]
    
    nb, nc, nt = x.shape

    probs = torch.ones_like(x) * r
    mask = torch.bernoulli(probs)
    # alternatively, the mask level could be the cumsum of the mask
    mask = mask.round().long().to(x.device)
    mask_levels = nc - mask.sum(dim=1) - 1

    # create a new mask, where all levels below the mask level are masked
    # shape (nb, nc, nt) where new_mask[i, CB:, t] = 1, CB = mask_level[i, t] 
    # mask = mask_levels[:, :, None] > torch.arange(nc)[None, None, :]
    mask = (mask_levels[:, None, :] < torch.arange(nc, device=x.device)[None, :, None]).long()

    ignore_levels = mask_levels + 1
    ignore_indices_mask = (ignore_levels[:, None, :] < torch.arange(nc, device=x.device)[None, :, None]).long()

    # for _b in range(nb):
    #     for _t in range(nt):
    #         for _c in range(nc):
    #             if mask[_b, _c, _t] == 1:
    #                 mask[_b, _c:, _t] = 1
    #                 ignore_indices_mask[_b, _c + 1:, _t] = 1
    #                 break
    
    return mask.long(), ignore_indices_mask.bool()


def better_cond_random_but_not_working(x: torch.Tensor, r:torch.Tensor):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device).float()
    
    r = _gamma(r)[:, None, None]
    
    nb, nc, nt = x.shape

    probs = torch.ones_like(x) * r
    mask = torch.bernoulli(probs)

    mask = mask.round().long().to(x.device)

    # there cannot be anything unmasked if there's an masked token
    # in the same timestep and below it 
    for i in range(nb):
        for j in range(nc):
            for t in range(nt):
                if mask[i, j, t] == 1:
                    mask[i, j:, t] = 1
                    break
    
    # an ignore indices mask, since we can truly only predict one token
    # per timestep
    ignore_indices = torch.zeros_like(x)
    for i in range(nb):
        for j in range(nc):
            for t in range(nt):
                if mask[i, j, t] == 1:
                    ignore_indices[i, j, t+1:] = 1
                    break
    return mask.int(), ignore_indices


@torch.jit.script_if_tracing
def linear_random(
    x: torch.Tensor,
    r: torch.Tensor,
):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device).float()
        r = r[:, None, None]

    probs = torch.ones_like(x).to(x.device).float()
    # expand to batch and codebook dims
    probs = probs.expand(x.shape[0], x.shape[1], -1)
    probs = probs * r

    mask = torch.bernoulli(probs)
    mask = mask.round().int()

    return mask

@torch.jit.script_if_tracing
def inpaint(x: torch.Tensor, n_prefix: int, n_suffix: int,):
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

@torch.jit.script_if_tracing
def periodic_mask(x: torch.Tensor, period: int,
                  width: int = 1, random_roll: bool = False,):
    mask = full_mask(x)
    if period == 0:
        return full_mask(x)

    if not isinstance(period, torch.Tensor):
        period = scalar_to_batch_tensor(period, x.shape[0])
    if period.ndim == 0:
        period = period[None]
        
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

def codebook_mask(mask: torch.Tensor, val1: int, val2: int = None):
    mask = mask.clone()
    mask[:, val1:, :] = 1
    # val2 = val2 or val1
    # vs = torch.linspace(val1, val2, mask.shape[1])
    # for t, v in enumerate(vs):
    #     v = int(v)
    #     mask[:, v:, t] = 1 

    return mask

@torch.jit.script_if_tracing
def mask_and(
    mask1: torch.Tensor, 
    mask2: torch.Tensor
):
    assert mask1.shape == mask2.shape, "masks must be same shape"
    return torch.min(mask1, mask2)

def drop_ones(mask: torch.Tensor, p: float):
    oldshp = mask.shape
    mask = mask.view(-1)

    # find ones idxs
    ones_idxs = torch.where(mask == 1)[0]
    # shuffle idxs
    ones_idxs_idxs = torch.randperm(len(ones_idxs))
    ones_idxs = ones_idxs[ones_idxs_idxs]
    # drop p% of ones
    ones_idxs = ones_idxs[:int(len(ones_idxs) * p)]
    # set those idxs to 0
    mask[ones_idxs] = 0

    mask = mask.view(oldshp)
    return mask


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
    onset_frame_idxs: torch.Tensor, 
    z: torch.Tensor,
    width: int = 1, 
):
    if len(onset_frame_idxs) == 0:
        print("no onsets detected")
    # print("onset_frame_idxs", onset_frame_idxs)
    # print("mask shape", z.shape)

    mask = torch.ones_like(z).int()
    for idx in onset_frame_idxs:
        mask[:, :, idx-width:idx+width] = 0

    return mask.int()

def tria_mask(
    codes: torch.Tensor, 
    min_amt: float = 0.1, 
    max_amt: float = 0.4,
):
    """ 
    unmasks a prefix of the codes tensor, 
    in the range provided
    """

    mask = full_mask(codes)
    nb, nc, nt = codes.shape
    for i in range(nb):
        amt = torch.rand(1) * (max_amt - min_amt) + min_amt
        amt = int(amt * nt)
        mask[i, :, :amt] = 0

    return mask






if __name__ == "__main__":
    sig = AudioSignal("assets/example.wav")
