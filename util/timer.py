
import torch
import time
# from logging import info as log_info

class _Timer:
    def __init__(self, enable, module_name):
        self.enable = enable
        self.module_name = module_name
        self._start = None
    def __enter__(self):
        if self.enable:
            torch.cuda.synchronize()
            self._start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.enable:
            torch.cuda.synchronize()
            self.run_time = time.time() - self._start
            print(f"{self.module_name} runtime: {self.run_time}s")
        return False


def timer(enable, module_name):
    return _Timer(enable, module_name)