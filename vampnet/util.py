import tqdm

import torch

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