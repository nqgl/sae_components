from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

from saeco.components.jumprelu.kernels_fns import rect
from saeco.components.penalties import Penalty

# import saeco.core as cl


class H_z_minus_thresh_fn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, z, thresh, kernel, eps):
        thresh = thresh.relu()
        gate = (z > thresh) & (z > 0)
        ctx.z = z
        ctx.save_for_backward(z, thresh)
        ctx.kernel = kernel
        ctx.eps = eps
        return torch.where(gate, 1, 0).to(z.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        z, thresh = ctx.saved_tensors
        eps = ctx.eps
        kernel = ctx.kernel
        thresh_grad = -1 / eps * kernel((z - thresh) / eps) * grad_output
        return None, thresh_grad, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        return torch.zeros_like(ctx.z)


def modified_H(n):
    class H_z_minus_thresh_modified_fn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, z, thresh, kernel, eps):
            thresh = thresh.relu()
            gate = (z > thresh) & (z > 0)
            ctx.save_for_backward(z, thresh)
            ctx.kernel = kernel
            ctx.eps = eps
            return torch.where(gate, 1, 0).to(z.dtype)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            z, thresh = ctx.saved_tensors
            eps = ctx.eps
            kernel = ctx.kernel
            thresh_grad = -1 / eps * kernel((z - thresh) / eps) * grad_output
            if n == 1:
                return -thresh_grad, thresh_grad, None, None
            if n == 2:
                return -thresh_grad, torch.where(z > 0, thresh_grad, 0), None, None
            if n == 3:
                return torch.where(z > 0, -thresh_grad, 0), thresh_grad, None, None
            if n == 4:
                return None, torch.where(z > 0, thresh_grad, 0), None, None

    return H_z_minus_thresh_modified_fn.apply


class HStep(nn.Module):
    def __init__(self, thresh, kernel, eps, modified_grad=False, exp=False):
        super().__init__()
        self.thresh = thresh
        self.kernel = kernel
        self.eps = eps
        self.exp = exp
        self.H = H_z_minus_thresh_fn.apply
        if modified_grad:
            self.H = modified_H(modified_grad)

    def forward(self, x):
        thresh = self.thresh
        if self.exp:
            thresh = thresh.exp()
        return self.H(x, thresh, self.kernel, self.eps)


class JumpReLU_fn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, z, thresh, kernel, eps):
        thresh = thresh.relu()
        gate = z > thresh
        ctx.save_for_backward(z, thresh, gate)
        ctx.eps = eps
        ctx.kernel = kernel
        return torch.where(gate, z, 0)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        z, thresh, gate = ctx.saved_tensors
        ctx.gate = gate
        eps = ctx.eps
        kernel = ctx.kernel
        thresh_grad = -thresh / eps * kernel((z - thresh) / eps) * grad_output
        z_grad = torch.where(gate, grad_output, 0)
        return z_grad, thresh_grad, None, None

    @staticmethod
    def jvp(ctx: Any, grad_in_z, grad_in_thresh, *etc: Any) -> Any:
        return torch.where(ctx.gate, grad_in_z, 0)


def jumprelu_modified(n):
    class JumpReLU_Modified_fn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, z, thresh, kernel, eps):
            thresh = thresh.relu()
            gate = z > thresh
            ctx.save_for_backward(z, thresh, gate)
            ctx.eps = eps
            ctx.kernel = kernel
            return torch.where(gate, z, 0)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            z, thresh, gate = ctx.saved_tensors
            eps = ctx.eps
            kernel = ctx.kernel
            thresh_grad = -thresh / eps * kernel((z - thresh) / eps) * grad_output
            z_grad = torch.where(gate, grad_output, 0)
            if n == 1:
                return z_grad - thresh_grad, thresh_grad, None, None
            if n == 2:
                return (
                    z_grad - thresh_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )
            if n == 3:
                return (
                    z_grad - torch.where(z > 0, thresh_grad, 0),
                    thresh_grad,
                    None,
                    None,
                )
            if n == 4:
                return (
                    z_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )

    return JumpReLU_Modified_fn.apply


class L0Penalty(Penalty):
    def __init__(
        self,
        thresh,
        eps,
        kernel=rect,
        modified_grad=False,
        exp=False,
        scale=1.0,
    ):
        super().__init__()
        self.thresh = thresh
        self.H = HStep(
            thresh=thresh,
            kernel=kernel,
            eps=eps,
            exp=exp,
            modified_grad=modified_grad,
        )

    def penalty(self, x, *, cache):
        return cache(self).H(x).sum(1).mean(0)


import saeco.core as cl
from saeco.components.features import FeaturesParam


class JumpReLU(cl.Module):
    def __init__(self, thresh, eps, kernel=rect, modified_jumprelu=False, exp=False):
        super().__init__()
        self.thresh = thresh
        self.kernel = kernel
        self.eps = eps
        self.jumprelu = JumpReLU_fn.apply
        if modified_jumprelu:
            self.jumprelu = jumprelu_modified(modified_jumprelu)
        self.exp = exp

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        thresh = self.thresh
        if self.exp:
            thresh = thresh.exp()
        return self.jumprelu(x, thresh, self.kernel, self.eps)

    @property
    def features(self):
        return dict(
            thresh=FeaturesParam(
                self.thresh,
                feature_index=0,
                feature_parameter_type="bias",
            )
        )
