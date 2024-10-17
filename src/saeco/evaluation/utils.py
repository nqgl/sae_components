from contextlib import contextmanager

import torch


@contextmanager
def fwad_safe_sdp():
    mem_sdp = torch.backends.cuda.mem_efficient_sdp_enabled()
    flash_sdp = torch.backends.cuda.flash_sdp_enabled()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        yield
    finally:
        torch.backends.cuda.enable_mem_efficient_sdp(mem_sdp)
        torch.backends.cuda.enable_flash_sdp(flash_sdp)
