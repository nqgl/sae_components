import torch
from torch.cuda.amp import custom_bwd, custom_fwd


def shrinkgrad_adjustment(errors, leniency, dd, b):
    return errors * (leniency * 2 / dd / b)


def modified_H2(n):
    n, leniency, dd = n

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
            b = z.shape[0]

            adjustment = shrinkgrad_adjustment(
                torch.where((~(z > thresh)) & (z > 0), z, 0),
                leniency=leniency,
                dd=dd,
                b=b,
            )
            if n == 0:
                return None, thresh_grad, None, None

            if n == 1:
                return -thresh_grad, thresh_grad, None, None
            if n == 2:
                return -thresh_grad, torch.where(z > 0, thresh_grad, 0), None, None
            if n == 3:
                return torch.where(z > 0, -thresh_grad, 0), thresh_grad, None, None
            if n == 4:
                return None, torch.where(z > 0, thresh_grad, 0), None, None

    return H_z_minus_thresh_modified_fn.apply


def jumprelu_modified2(n):
    n, leniency, dd = n

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
            b = z.shape[0]
            adjustment = shrinkgrad_adjustment(
                torch.where((z < thresh) & (z > 0), z, 0),
                leniency=leniency,
                dd=dd,
                b=b,
            )
            # z_grad = torch.where(
            #     gate, grad_output, 0
            # )  # avoid double-counting the adjustment
            # grad_output = grad_output + adjustment
            z_grad = torch.where(
                z > thresh, grad_output, 0
            )  # avoid double-counting the adjustment
            # grad_output = grad_output + adjustment
            if n == 8:
                z_grad = (
                    torch.where(z > 0, grad_output, 0)
                    + shrinkgrad_adjustment(
                        torch.where((z < thresh) & (z > thresh / 2), z, 0),
                        leniency=leniency,
                        dd=dd,
                        b=b,
                    )
                    * 0.5
                )
                z_grad = torch.where(z > thresh, z_grad, z_grad * 0.1)

            thresh_grad = (
                -thresh / eps * kernel((z - thresh) / eps) * (grad_output + adjustment)
            )
            zthresh_grad = (
                thresh
                / eps
                * kernel((z - thresh) / eps)
                * (
                    grad_output
                    + shrinkgrad_adjustment(
                        torch.where((z < thresh) & (z > 0), thresh, 0),
                        leniency=leniency,
                        dd=dd,
                        b=b,
                    )
                )
            )
            # thresh_mul = -thresh / eps * kernel((z - thresh) / eps)

            # zthresh_grad = thresh_grad - adjustment
            if n == 1:
                return (
                    z_grad - thresh_grad,
                    thresh_grad,
                    None,
                    None,
                )
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
            if n == 5:
                return (
                    z_grad + zthresh_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )
            if n == 6:
                return (
                    z_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )
            if n == 7:
                return (
                    z_grad - thresh_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )
            if n == 8:
                return (
                    z_grad,
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )
            if n == 9:
                return (
                    z_grad - torch.where(z < thresh, thresh_grad, 0),
                    torch.where(z > 0, thresh_grad, 0),
                    None,
                    None,
                )

    return JumpReLU_Modified_fn.apply


def modify_modified():
    import saeco.components.jumprelu.jumprelu_fn as jumprelu_fn

    jumprelu_fn.jumprelu_modified = jumprelu_modified2
    jumprelu_fn.modified_H = modified_H2
