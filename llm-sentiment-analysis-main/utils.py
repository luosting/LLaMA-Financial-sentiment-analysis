import time
import torch
from contextlib import contextmanager

@contextmanager
def measure_peak_memory(tag):
    torch.cuda.reset_max_memory_allocated()
    yield
    peak_memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"Peak memory usage ({tag}): {peak_memory_usage:.1f} MB")

@contextmanager
def set_default_dtype(dtype):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(original_dtype)

def dist_print(msg):
    import time
    torch.distributed.barrier()
    time.sleep(torch.distributed.get_rank()/100.0)
    print(msg)
