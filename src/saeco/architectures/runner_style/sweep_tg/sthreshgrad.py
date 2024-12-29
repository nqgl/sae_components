import torch
import torch.nn as nn
from saeco.components.penalties.l0targeter import L0Targeting
from saeco.sweeps import SweepableConfig
from torch.amp import custom_bwd, custom_fwd


def shrinkgrad_adjustment(errors, leniency, dd, b):
    return errors * (leniency * 2 / dd / b)


# class GTConfig(SweepableConfig):
#     gate_pres: torch.Tensor
#     threshes: torch.Tensor
#     gate_posts: torch.Tensor
#     leniency: float
#     dd: float
#     loss_coeff: float


def sig_grad_window(x):
    sig = x.sigmoid()
    return sig * (1 - sig)


windows = {"sig": sig_grad_window}


def GT2(grad_window=sig_grad_window):
    class GT2(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda")
        def forward(ctx, gate_pre, gate_post, noise, mag, leniency, d_data=768):
            gate = gate_pre > 0
            ctx.save_for_backward(gate_pre, mag, gate)
            ctx.gate_post = gate_post
            ctx.leniency = leniency
            ctx.d_data = d_data
            ctx.noise = noise
            return gate.float()

        @staticmethod
        @custom_bwd(device_type="cuda")
        def backward(ctx: torch.Any, grad_output):

            gate_pre, mag, gate = ctx.saved_tensors
            gate_post = ctx.gate_post
            leniency = ctx.leniency
            noise = ctx.noise

            offgrad_mask = (~gate) & (grad_output != 0) & (noise > 0)
            grad_output = torch.where(mag != 0, grad_output / mag, 0)
            d_data = ctx.d_data
            batch_size = gate_pre.shape[0]
            off_adjustment = shrinkgrad_adjustment(
                mag,
                leniency=leniency,
                dd=d_data,
                b=batch_size,
            )
            grad_gate = (noise < 0) & (gate_post > 0)
            adjustment = shrinkgrad_adjustment(
                -noise,
                leniency=leniency,
                dd=d_data,
                b=batch_size,
            )
            # grad_output = torch.where(grad_gate, , 0)
            # grad_output = (
            #     grad_output + adjustment
            # )  # TODO should +adj be moved to the .where below?
            #    #  seems like clear yes, so trying that
            # return ( this is interesting kinda works actually so should try it out more
            # or like, it but better
            #     (grad_output + adjustment) * grad_window(gate_pre),
            #     None,
            #     None,
            #     None,
            #     None,
            # )

            out = torch.where(grad_gate, grad_output + adjustment, 0) + torch.where(
                offgrad_mask,
                grad_output + adjustment,
                0,
            )
            return (
                out * grad_window(gate_pre),
                None,
                None,
                None,
                None,
            )

    return GT2.apply


def gate(
    mag,
    gate_pres,
    GT_fn,
    exp_mag=False,
    training=True,
    uniform_noise=False,
    noise_mult=1,
    leniency=1,
    penalty_fn=None,
):
    future_gate_post = torch.ones_like(mag)
    mag = mag.relu() if not exp_mag else mag.exp()
    gate = None
    p_noise = 0.2
    noise_mask = torch.rand_like(mag) < p_noise

    noise = (
        torch.where(
            (mag > 0),
            (0.5 - torch.rand_like(mag)) if uniform_noise else torch.randn_like(mag),
            0,
        )
        * noise_mult
        if training
        else torch.zeros_like(mag)
    ).cuda()
    for gate_pre in gate_pres:
        g = GT_fn(gate_pre, future_gate_post, noise, mag, leniency)
        if gate is None:
            gate = g
        else:
            gate = gate * g
    with torch.no_grad():
        future_gate_post[:] = gate[:]
        noise[:] = torch.where((gate > 0) | noise_mask, noise, 0)
    if penalty_fn is not None:
        penalty_fn(gate * mag.detach())
    out = gate * mag + noise
    return out


import saeco.core as cl
from saeco.components.features import FeaturesParam
from saeco.initializer import Initializer


class Config(SweepableConfig):
    pre_bias: bool = False
    uniform_noise: bool = True
    noise_mult: float = 0.1
    exp_mag: bool = True
    mag_weights: bool = False
    window_fn: str = "sig"
    decay_l1: bool = True
    leniency_targeting: bool = False
    leniency: float = 1


class BinaryEncoder(cl.Module):
    def __init__(
        self,
        cfg: Config,
        init: Initializer,
        penalty=None,
        apply_targeting_externally=False,
        signed_mag: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.mag = init._encoder.new_bias()
        self.mag.data -= 1
        # self.mag.data += 2
        self.gate = init.encoder
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0002 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = GT2(windows[cfg.window_fn])
        self.apply_targeting_externally = apply_targeting_externally
        self.signed_mag = signed_mag
        self.penalty = penalty

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        if (
            (not self.cfg.leniency_targeting)
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        ):

            self.targeting.value = cache._ancestor.trainstep / 5000
        mag = self.mag.unsqueeze(0).expand(x.shape[0], -1)
        out = gate(
            mag.abs() if self.signed_mag else mag,
            [self.gate(x)],
            GT_fn=self.GT_fn,
            exp_mag=not self.signed_mag,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=self.targeting.value
            * (
                1
                if not cache._ancestor.has.trainer
                else cache._ancestor.trainer.cfg.schedule.lr_scale(
                    cache._ancestor.trainstep
                )
            ),
            penalty_fn=cache(self).penalty if self.penalty is not None else None,
        )
        if not self.apply_targeting_externally:
            cache(self).targeting(out)
        out = out * torch.sign(mag) if self.signed_mag else out
        return out

    @property
    def features(self):
        return {"mag": FeaturesParam(self.mag, 0, "bias")}


class GTTest(cl.Module):
    def __init__(self, cfg: Config, init: Initializer, targeting=True):
        super().__init__()
        self.cfg = cfg
        self.mag = nn.Linear(init.d_data, init.d_dict)
        # self.mag.data += 2
        self.gate = init.encoder
        self.gate.weight.grad
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = GT2(windows[cfg.window_fn])

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        out = gate(
            self.mag(x),
            [self.gate(x)],
            GT_fn=self.GT_fn,
            exp_mag=self.cfg.exp_mag,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=self.targeting.value,
        )
        cache(self).targeting(out)
        return out

    @property
    def features(self):
        return {
            "mag": FeaturesParam(self.mag.weight, 0, "other"),
            "mag_bias": FeaturesParam(self.mag.bias, 0, "bias"),
        }


class GTMulti(cl.Module):
    def __init__(
        self, cfg: Config, init: Initializer, d_in_override=None, targeting=True
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.mag_weights:
            self.mag = init._encoder.new_bias()
            self.mag.data -= 1
        # self.mag.data += 2
        self.gate = init.encoder
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )
        self.GT_fn = GT2(windows[cfg.window_fn])

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.leniency = ...
        cache.leniency = self.targeting.value
        if (
            (not self.cfg.leniency_targeting)
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        ):
            self.targeting.value = cache._ancestor.trainstep / 5000
        out = gate(
            self.mag.unsqueeze(0).expand(x.shape[0], -1),
            [self.gate(x)],
            GT_fn=self.GT_fn,
            exp_mag=True,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            leniency=self.targeting.value,
        )
        cache(self).targeting(out)
        return out

    @property
    def features(self):
        return {"mag": FeaturesParam(self.mag, 0, "bias")}
