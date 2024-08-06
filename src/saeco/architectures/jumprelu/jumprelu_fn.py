import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

from saeco.components.penalties import Penalty

# import saeco.core as cl


class H_z_minus_thresh_fn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, z, thresh, kernel, eps):

        thresh = thresh.relu()
        gate = z > thresh
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


class HStep(nn.Module):
    def __init__(self, thresh, kernel, eps):
        super().__init__()
        self.thresh = thresh
        self.kernel = kernel
        self.eps = eps

    def forward(self, x):
        return H_z_minus_thresh_fn.apply(x, self.thresh, self.kernel, self.eps)


def rect(x: torch.Tensor) -> torch.Tensor:
    return x.abs() < 0.5


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
        eps = ctx.eps
        kernel = ctx.kernel
        thresh_grad = -thresh / eps * kernel((z - thresh) / eps) * grad_output
        z_grad = torch.where(gate, grad_output, 0)
        return z_grad, thresh_grad, None, None


class L0Penalty(Penalty):
    def __init__(
        self,
        thresh,
        eps,
        kernel=rect,
        scale=1.0,
    ):
        super().__init__()
        self.thresh = thresh
        self.H = HStep(
            thresh=thresh,
            kernel=kernel,
            eps=eps,
        )

    def penalty(self, x, *, cache):
        return cache(self).H(x).sum(1).mean(0)


import saeco.core as cl
from saeco.components.features import FeaturesParam


class JumpReLU(cl.Module):
    def __init__(self, thresh, eps, kernel=rect):
        super().__init__()
        self.thresh = thresh
        self.kernel = kernel
        self.eps = eps

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return JumpReLU_fn.apply(x, self.thresh, self.kernel, self.eps)

    @property
    def features(self):
        return dict(
            thresh=FeaturesParam(
                self.thresh,
                feature_index=0,
                fptype="bias",
            )
        )
