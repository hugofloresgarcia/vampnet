import tqdm
import random
import time

import torch
import numpy as np
from einops import rearrange

def seed(random_seed, set_cudnn=False):
    """
    Seeds all random states with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``torch``
    random generators.
    For full reproducibility, two further options must be set
    according to the torch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html
    To do this, ``set_cudnn`` must be True. It defaults to
    False, since setting it to True results in a performance
    hit.

    Args:
        random_seed (int): integer corresponding to random seed to
        use.
        set_cudnn (bool): Whether or not to set cudnn into determinstic
        mode and off of benchmark mode. Defaults to False.
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flip_coin(prob):
    return torch.rand(1).item() < prob

def first_dict_value(d):
    return next(iter(d.values()))

def first_dict_key(d):
    return next(iter(d.keys()))

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

class Timer:
    
    def __init__(self):
        self.times = {}
    
    def tick(self, name: str):
        self.times[name] = time.time()
    
    def tock(self, name: str):
        toc = time.time() - self.times[name]
        print(f"{name} took {toc} seconds")
        return toc
    
    def __str__(self):
        return str(self.times)

def buffer_plot_and_get(fig, **kwargs):
    import io
    import PIL
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    return PIL.Image.open(buf)