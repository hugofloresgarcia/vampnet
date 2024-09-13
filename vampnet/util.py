import tqdm

import torch
from einops import rearrange

def scalar_to_batch_tensor(x, batch_size):
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
    return rearrange(tokens, "b c t -> b (t c)")

def codebook_unflatten(flat_tokens: torch.Tensor, n_c: int = None):
    """
    unflatten a sequence of tokens from (batch, codebook * time) to (batch, codebook, time)
    """
    tokens = rearrange(flat_tokens, "b (t c) -> b c t", c=n_c)
    return tokens
