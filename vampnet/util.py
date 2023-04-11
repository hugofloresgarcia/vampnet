import tqdm
# import pathos

def process_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    """
    from concurrent.futures import ProcessPoolExecutor
    if iterables and "chunksize" not in tqdm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len > 1000:
            from warnings import warn
            warn("Iterable length %d > 1000 but `chunksize` is not set."
                 " This may seriously degrade multiprocess performance."
                 " Set `chunksize=1` or more." % longest_iterable_len,
                 TqdmWarning, stacklevel=2)
    if "lock_name" not in tqdm_kwargs:
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs["lock_name"] = "mp_lock"
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)


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