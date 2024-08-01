from saeco.components.wrap import WrapsModule


import torch.nn as nn


class ClipGrad(WrapsModule):
    def __init__(self, module, max_norm=1):
        super().__init__(module)
        self.max_norm = max_norm

    def post_backward_hook(self):
        nn.utils.clip_grad_norm_(self.parameters(), float(self.max_norm))
