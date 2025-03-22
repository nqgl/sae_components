from saeco.components.wrap import WrapsModule
from saeco.components.type_acc_methods import post_backward_hook


import torch.nn as nn


class ClipGrad(WrapsModule):
    def __init__(self, module, max_norm=1):
        super().__init__(module)
        self.max_norm = max_norm

    def post_backward_hook(self):
        nn.utils.clip_grad_norm_(self.parameters(), float(self.max_norm))


class ClipGradMixin(nn.Module):
    _clip_grad_mixin_max_norm: float = 1.0

    @post_backward_hook
    def clip_parameter_grads(self):
        nn.utils.clip_grad_norm_(
            self.parameters(), float(self._clip_grad_mixin_max_norm)
        )

    @classmethod
    def with_max_norm(cls, max_norm: float) -> type:
        class ClipGradMixinParameterized(cls):
            _clip_grad_mixin_max_norm = max_norm

        return ClipGradMixinParameterized
