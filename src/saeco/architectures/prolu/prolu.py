import torch
from torch.cuda.amp import custom_bwd, custom_fwd


class ProLU(torch.autograd.Function):
    STE: torch.autograd.Function
    ReLU: torch.autograd.Function

    @staticmethod
    @custom_fwd
    def forward(ctx, m, b):
        gate = (m + b > 0) & (m > 0)
        ctx.save_for_backward(m, gate)
        return torch.where(gate, m, 0)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "This method should be overridden by a subclass of ProLU to provide a backward implementation."
        )


class ProLU_ReLU(ProLU):
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        m, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_m, grad_b = gated_grad.clone(), gated_grad.clone()
        return grad_m, grad_b, None


class ProLU_STE(ProLU):
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        m, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_b = gated_grad * m
        grad_m = gated_grad + grad_b.clone()
        return grad_m, grad_b, None


from saeco.sweeps import SweepableConfig


class ProLUConfig(SweepableConfig):
    b_ste: float
    m_ste: float
    m_gg: float


def paramaterized_prolu(cfg: ProLUConfig):
    class ProLUp(ProLU):
        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            m, gate = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)
            gg = gated_grad
            ggm = gated_grad * m

            grad_b_ste = ggm
            grad_m_ste = gg * cfg.m_gg + ggm
            grad_b_0 = gg
            grad_m_0 = gg
            grad_b = grad_b_0.lerp(grad_b_ste, cfg.b_ste)
            grad_m = grad_m_0.lerp(grad_m_ste, cfg.m_ste)
            return grad_m, grad_b

    return ProLUp.apply


from saeco.components.features.features_param import FeaturesParam

from typing import Callable
from saeco.core import Module


class PProLU(Module):
    def __init__(self, prolu: ProLUConfig | Callable, d_bias):
        super().__init__()
        if isinstance(prolu, ProLUConfig):
            # self.cfg = prolu
            self.prolu = paramaterized_prolu(prolu)
        else:
            self.prolu = prolu
        if isinstance(d_bias, int):
            self.bias = torch.nn.Parameter(torch.zeros(d_bias))
        elif isinstance(d_bias, torch.nn.Parameter):
            self.bias = d_bias
        else:
            raise ValueError(f"Invalid bias type {type(d_bias)}")

    def forward(self, m, *, cache=None):
        return self.prolu(m, self.bias)

    @property
    def features(self) -> dict[str, FeaturesParam]:
        return {"bias": FeaturesParam(self.bias, feature_index=0, fptype="bias")}


ProLU.STE = ProLU_STE
ProLU.ReLU = ProLU_ReLU


def prolu_ste(m, b):
    return ProLU_STE.apply(m, b)


def prolu_relu(m, b):
    return ProLU_ReLU.apply(m, b)


class ThreshSTE(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x):
        gate = x > 0
        ctx.save_for_backward(x, gate)
        return gate.float()

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        return gated_grad


class Thresh(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x):
        gate = x > 0
        ctx.save_for_backward(x, gate)
        return gate.float()

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "This method should be overridden by a subclass of Thresh to provide a backward implementation."
        )


class ThreshSTE(Thresh):
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output
        # x, gate = ctx.saved_tensors
        # gated_grad = torch.where(gate, grad_output, 0)


def thresh_from_bwd(bwd_fn: callable):
    class CustThresh(Thresh):
        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            x, gate = ctx.saved_tensors
            return bwd_fn(x) * grad_output

    return CustThresh.apply


def prolu_ste_from_thresh(thresh: callable):
    def prolu(m, b):
        return m * thresh(m + b) * thresh(m)

    return prolu
