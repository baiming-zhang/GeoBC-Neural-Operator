"""Shared GPU lock and an elapsed-time/memory budget."""
import time
from collections import defaultdict
from pathlib import Path

class Budget:
    def __init__(self, seconds=900, memory_bytes=30_500_000_000):
        self.deadline = time.time() + seconds
        self.monotonic_deadline = time.monotonic() + seconds
        self.memory_bytes = memory_bytes

    def __call__(self):
        if min(self.deadline-time.time(), self.monotonic_deadline-time.monotonic()) <= 0:
            raise TimeoutError('Time budget exhausted')
        import torch
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            if total-free > self.memory_bytes:
                raise MemoryError('GPU memory budget exceeded')

class GPUQueue:
    def __init__(self, path='runs/.gpu.lock'):
        self.path = Path(path)
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+')
        fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self
    def __exit__(self, *args):
        self.file.close()

class CoarseLedger:
    def __init__(self, torch, guard):
        self.torch, self.guard = torch, guard
        self.data = defaultdict(float)
    def sync(self):
        self.torch.cuda.synchronize()
    def check(self):
        self.guard()
    def account(self):
        self.guard()
    def check_memory(self):
        self.guard()
