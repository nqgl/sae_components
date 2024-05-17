import torch.nn as nn


class NegBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        assert isinstance(bias, nn.Parameter)
        self.bias = bias

    def forward(self, x):
        return x - self.bias
