import torch.nn as nn

from saeco.components.type_acc_methods import post_backward_hook
from saeco.components.wrap import WrapsModule


class ClipGrad(WrapsModule[nn.Module]):
    """Clip ``self.parameters()`` to ``max_norm`` after each backward pass."""

    def __init__(self, module: nn.Module, max_norm: float = 1.0):
        super().__init__(module)
        self._self_max_norm = max_norm

    @post_backward_hook
    def clip_parameter_grads(self) -> None:
        # `self.parameters()` proxies through wrapt to `self.__wrapped__.parameters()`.
        nn.utils.clip_grad_norm_(self.parameters(), float(self._self_max_norm))
