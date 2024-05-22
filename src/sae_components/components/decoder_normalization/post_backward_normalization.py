import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from sae_components.components.sae_cache import SAECache
from torch import Tensor
from jaxtyping import Float

import sae_components.core as cl


def post_backward(module: nn.Module):
    if hasattr(module, "post_backward_hook"):
        module.post_backward_hook()


def post_step(module: nn.Module):
    if hasattr(module, "post_step_hook"):
        return module.post_step_hook()
