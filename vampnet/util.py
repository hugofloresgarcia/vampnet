import tqdm

import torch
import numpy as np
from einops import rearrange

def flip_coin(prob):
    return torch.rand(1).item() < prob

def first_dict_value(d):
    return next(iter(d.values()))

@torch.jit.script
def scalar_to_batch_tensor(x: int | float, batch_size: int):
    # torchscript weirdness
    if isinstance(x, int):
        return torch.tensor(x).repeat(batch_size)
    else:
        return torch.tensor(x).repeat(batch_size)

def parallelize(
        fn, 
        *iterables,
        parallel: str = "thread_map",
        **kwargs
    ):
    if parallel == "thread_map":
        from tqdm.contrib.concurrent import thread_map
        return thread_map(
            fn, 
            *iterables, 
            **kwargs
        )
    elif parallel == "process_map":
        from tqdm.contrib.concurrent import process_map
        return process_map(
            fn, 
            *iterables, 
            **kwargs
        )
    elif parallel == "single":
        return [fn(x) for x in tqdm.tqdm(*iterables)]
    else:
        raise ValueError(f"parallel must be one of 'thread_map', 'process_map', 'single', but got {parallel}")
    
def codebook_flatten(tokens: torch.Tensor):
    """ 
    flatten a sequence of tokens from (batch, codebook, time) to (batch, codebook * time)
    """
    # return rearrange(tokens, "b c t -> b (t c)")/
    return tokens.permute(0, 2, 1).flatten(1, 2)

def codebook_unflatten(flat_tokens: torch.Tensor, n_c: int = None):
    """
    unflatten a sequence of tokens from (batch, codebook * time) to (batch, codebook, time)
    """
    # tokens = rearrange(flat_tokens, "b (t c) -> b c t", c=n_c)
    # return tokens
    return flat_tokens.view(flat_tokens.shape[0], -1, n_c).permute(0, 2, 1)
