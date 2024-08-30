# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
from saeco.trainer import Trainable
from safetensors.torch import save_file, load_file
import torch


# tensor = torch.randn(2**11, 2**10, 2**10)

# save_file({"tensor": tensor}, "tensor.pt")
tensor = load_file("tensor.pt")["tensor"]
input(tensor.is_shared())
tensor.share_memory_()
print("shared")
print()
