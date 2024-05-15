# by Glen Taggart @nqgl
import torch
import torch.nn.functional as F
from typing import Optional


class PositiveGradthruIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(min=0)


class NegativeGradthruIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(max=0)


def undying_relu(
    x, l=0.01, k=1, l_mid_neg=None, l_low_neg=0, l_low_pos=None, leaky=False
):
    """
    Compute undying ReLU activation function.
    Behaves like a normal ReLU on the forward pass, but allows gradients to pass through
    negative values during the backward pass.

    Args:
        x (torch.Tensor): Input tensor.
        l (float, optional): Leakage parameter for negative values. Stand-in for l_mid_pos if leaky=False. Defaults to 0.01.
        k (float, optional): Threshold parameter. Defaults to 1. Value ranges are defined as
            low < -k < mid < 0
        l_mid_neg (float, optional): Leakage parameter for negative values in the mid-range. Defaults to l.
        l_low_neg (float, optional): Leakage parameter for negative values in the low range. Defaults to 0.
        l_low_pos (float, optional): Leakage parameter for positive values in the low range. Defaults to l.
        leaky (bool, optional): Whether to use leaky gradients, in which case only l must be defined.

    Returns:
        torch.Tensor: Output tensor after applying the undying ReLU activation function.
    """
    if l_mid_neg is None:
        l_mid_neg = l
    if l_low_pos is None:
        l_low_pos = l
    if leaky:
        k = 0
        l_low_neg, l_mid_neg, l_low_pos = l, l, l
    l_mid_pos = l
    y_forward = F.relu(x)
    y_backward1 = x * (x > 0)
    y_backward2_pos = (
        l_mid_pos
        * NegativeGradthruIdentityFunction.apply(x)
        * (torch.logical_and(x <= 0, x > -k))
    )
    y_backward2_neg = (
        l_mid_neg
        * PositiveGradthruIdentityFunction.apply(x)
        * (torch.logical_and(x <= 0, x > -k))
    )
    y_backward3_pos = l_low_pos * NegativeGradthruIdentityFunction.apply(x) * (x <= -k)
    y_backward3_neg = l_low_neg * PositiveGradthruIdentityFunction.apply(x) * (x <= -k)
    y_backward = (
        y_backward1
        + y_backward2_pos
        + y_backward2_neg
        + y_backward3_pos
        + y_backward3_neg
    )
    return y_backward + (y_forward - y_backward).detach()
