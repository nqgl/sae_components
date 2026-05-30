"""
Gated Binary Encoder with Custom Straight-Through Estimator

This module implements a sparse autoencoder encoder that uses binary gating
with a learned magnitude per feature. The key innovation is a custom backward
pass that:
1. Uses a gradient window (sigmoid derivative) to smooth the STE gradient
2. Applies "leniency" adjustments to encourage features to turn on/off appropriately
3. Handles off-features that receive gradient signal specially
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp.autocast_mode import custom_bwd, custom_fwd

import saeco.core as cl
from saeco.architecture.sae import SAE
from saeco.architecture.sae_architecture import Architecture
from saeco.components.features import FeaturesParam
from saeco.components.penalties.l0targeter import L0Targeting
from saeco.initializer import Initializer
from saeco.sweeps import SweepableConfig

# =============================================================================
# Gradient Window Functions
# =============================================================================


def sigmoid_grad_window(x: Tensor) -> Tensor:
    """
    Derivative of sigmoid: sig(x) * (1 - sig(x))

    Used to weight the STE gradient - gives smooth falloff for pre-activations
    far from the threshold, focusing gradient signal near the decision boundary.
    """
    sig = x.sigmoid()
    return sig * (1 - sig)


GradWindowFn = Callable[[Tensor], Tensor]

GRAD_WINDOWS: dict[str, GradWindowFn] = {
    "sig": sigmoid_grad_window,
}


# =============================================================================
# Custom Autograd Function for Gated STE
# =============================================================================


def _compute_leniency_adjustment(
    values: Tensor,
    leniency: float,
    d_data: int,
    batch_size: int,
) -> Tensor:
    """
    Compute gradient adjustment scaled by leniency and normalized by dimensions.

    This adjustment encourages features to change their gate state by providing
    an additional gradient signal proportional to the feature's magnitude/noise.
    """
    return values * (leniency * 2 / d_data / batch_size)


class GatedSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for binary gating with custom gradient handling.

    Forward: Returns binary gate (1 where gate_pre > 0, else 0)
    Backward: Custom gradient that:
        - Applies gradient window (sigmoid derivative) to focus on decision boundary
        - Adds leniency-scaled adjustments for features that should turn on/off
        - Handles off-features receiving gradient specially (via noise > 0 mask)
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        gate_pre: Tensor,
        noise: Tensor,
        mag: Tensor,
        leniency: float,
        d_data: int,
        grad_window: GradWindowFn,
    ) -> Tensor:
        gate = gate_pre > 0

        ctx.save_for_backward(gate_pre, mag, gate, noise)
        ctx.leniency = leniency
        ctx.d_data = d_data
        ctx.grad_window = grad_window

        return gate.float()

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: Tensor):
        gate_pre, mag, gate, noise = ctx.saved_tensors
        batch_size = gate_pre.shape[0]

        # Normalize gradient by magnitude (avoid div by zero)
        grad_normalized = torch.where(mag != 0, grad_output / mag, 0.0)

        adjustment = _compute_leniency_adjustment(
            -noise,
            leniency=ctx.leniency,
            d_data=ctx.d_data,
            batch_size=batch_size,
        )

        # Two gradient paths:
        # 1. grad_gate_mask: Features that are ON but noise suggests they should turn off
        #    (noise < 0 means we sampled below threshold, gate_post > 0 means still active)
        # 2. off_grad_mask: Features that are OFF but receiving gradient and have positive noise
        #    (these are candidates to turn on)
        grad_gate_mask = (noise < 0) & (gate)  # Note: gate_post was always == gate
        off_grad_mask = (~gate) & (grad_output != 0) & (noise > 0)

        grad_gate_pre = torch.where(
            grad_gate_mask, grad_normalized + adjustment, 0.0
        ) + torch.where(off_grad_mask, grad_normalized + adjustment, 0.0)

        # Apply gradient window to focus signal near decision boundary
        grad_gate_pre = grad_gate_pre * ctx.grad_window(gate_pre)

        # Return grads for: gate_pre, noise, mag, leniency, d_data, grad_window
        return grad_gate_pre, None, None, None, None, None


# =============================================================================
# Gating Logic
# =============================================================================


class GateConfig(SweepableConfig):
    """Configuration for the gating mechanism."""

    pre_bias: bool = False
    uniform_noise: bool = True
    noise_mult: float = 0.1
    exp_mag: bool = True
    mag_weights: bool = False
    window_fn: Literal["sig"] = "sig"
    decay_l1: bool = True
    leniency_targeting: bool = False
    leniency: float = 1.0


def apply_gating(
    mag: Tensor,
    gate_pre: Tensor,
    *,
    grad_window: GradWindowFn,
    d_data: int,
    leniency: float,
    training: bool,
    uniform_noise: bool = True,
    noise_mult: float = 0.1,
    exp_mag: bool = True,
    penalty_fn: Callable[[Tensor], None] | None = None,
    p_noise: float = 0.2,
) -> Tensor:
    """
    Apply gated activation with noise injection and custom STE backward.

    Args:
        mag: Magnitude values for each feature (pre-activation)
        gate_pre: Pre-activation gate values from encoder
        grad_window: Function to weight gradients by distance from threshold
        d_data: Data dimensionality (for gradient normalization)
        leniency: Scale factor for gradient adjustments (higher = more forgiving)
        training: Whether in training mode (enables noise injection)
        uniform_noise: Use uniform vs normal noise distribution
        noise_mult: Scale factor for injected noise
        exp_mag: If True, apply exp() to mag; else relu()
        penalty_fn: Optional callback for L0/sparsity penalty computation
        p_noise: Probability of keeping noise for off-features

    Returns:
        Gated output: gate * magnitude + noise
    """
    # Transform magnitude to positive values
    mag_activated = mag.exp() if exp_mag else mag.relu()

    # Generate noise (only during training, only where mag > 0)
    if training:
        noise_raw = (
            (0.5 - torch.rand_like(mag_activated))
            if uniform_noise
            else torch.randn_like(mag_activated)
        )
        noise = torch.where(mag_activated > 0, noise_raw, 0.0) * noise_mult
    else:
        noise = torch.zeros_like(mag_activated)

    # Ensure noise is on correct device
    noise = noise.to(mag_activated.device)

    # Apply custom STE gating
    gate = GatedSTE.apply(
        gate_pre,
        noise,
        mag_activated,
        leniency,
        d_data,
        grad_window,
    )

    # Sparsify noise: only keep noise where gate is on OR randomly sampled
    with torch.no_grad():
        noise_mask = torch.rand_like(mag_activated) < p_noise
        noise = torch.where((gate > 0) | noise_mask, noise, 0.0)

    # Compute penalty on gated magnitudes (detached) if provided
    if penalty_fn is not None:
        penalty_fn(gate * mag_activated.detach())

    return gate * mag_activated + noise


# =============================================================================
# Encoder Modules
# =============================================================================


class BinaryEncoder(cl.Module):
    """
    Gated sparse encoder with learned per-feature magnitudes.

    Each feature has:
    - A magnitude parameter (transformed via exp or kept signed)
    - A gate determined by encoder(x) > 0

    The custom backward pass encourages appropriate sparsity via leniency-scaled
    gradient adjustments.
    """

    def __init__(
        self,
        cfg: GateConfig,
        init: Initializer,
        penalty: cl.Module | None = None,
        apply_targeting_externally: bool = False,
        signed_mag: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.signed_mag = signed_mag
        self.apply_targeting_externally = apply_targeting_externally
        self.penalty = penalty

        # Learnable magnitude per feature (initialized below 0 so exp(mag) < 1)
        self.mag = init._encoder.new_bias()
        self.mag.data -= 1

        # Gate encoder: maps input -> pre-activation gate values
        self.gate_encoder = init.encoder

        # L0 targeting for sparsity control
        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.005 if cfg.leniency_targeting else 0.0,
        )

        self._grad_window = GRAD_WINDOWS[cfg.window_fn]

    def forward(self, x: Tensor, *, cache: cl.Cache, **kwargs) -> Tensor:
        # Expose leniency in cache for logging/debugging
        cache.leniency = self.targeting.value

        # Warmup leniency over first 5k steps if not using adaptive targeting
        if self._should_warmup_leniency(cache):
            self.targeting.value = cache._ancestor.trainstep / 5000

        # Expand magnitude to batch dimension
        batch_size = x.shape[0]
        mag = self.mag.unsqueeze(0).expand(batch_size, -1)

        # Handle signed magnitude mode
        if self.signed_mag:
            mag_input = mag.abs()
            exp_mag = False
        else:
            mag_input = mag
            exp_mag = True

        # Compute effective leniency (scaled by LR schedule if available)
        effective_leniency = self._compute_effective_leniency(cache)

        # Apply gating
        out = apply_gating(
            mag_input,
            self.gate_encoder(x),
            grad_window=self._grad_window,
            d_data=x.shape[-1],
            leniency=effective_leniency,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            exp_mag=exp_mag,
            penalty_fn=cache(self).penalty if self.penalty is not None else None,
        )

        # Apply L0 targeting (unless handled externally)
        if not self.apply_targeting_externally:
            cache(self).targeting(out)

        # Restore sign if using signed magnitude
        if self.signed_mag:
            out = out * torch.sign(self.mag)

        return out

    def _should_warmup_leniency(self, cache: cl.Cache) -> bool:
        return (
            not self.cfg.leniency_targeting
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        )

    def _compute_effective_leniency(self, cache: cl.Cache) -> float:
        base_leniency = self.targeting.value
        if cache._ancestor.has.trainer:
            lr_scale = cache._ancestor.trainer.cfg.schedule.lr_scale(
                cache._ancestor.trainstep
            )
            return base_leniency * lr_scale
        return base_leniency

    @property
    def features(self) -> dict[str, FeaturesParam]:
        return {"mag": FeaturesParam(self.mag, 0, "bias")}


class GTTest(cl.Module):
    """
    Test variant with learned magnitude as a linear layer (input-dependent magnitude).
    """

    def __init__(self, cfg: GateConfig, init: Initializer, targeting: bool = True):
        super().__init__()
        self.cfg = cfg

        self.mag_layer = nn.Linear(init.d_data, init.d_dict)
        self.gate_encoder = init.encoder

        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )

        self._grad_window = GRAD_WINDOWS[cfg.window_fn]

    def forward(self, x: Tensor, *, cache: cl.Cache, **kwargs) -> Tensor:
        cache.leniency = self.targeting.value

        out = apply_gating(
            self.mag_layer(x),
            self.gate_encoder(x),
            grad_window=self._grad_window,
            d_data=x.shape[-1],
            leniency=self.targeting.value,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            exp_mag=self.cfg.exp_mag,
        )

        cache(self).targeting(out)
        return out

    @property
    def features(self) -> dict[str, FeaturesParam]:
        return {
            "mag": FeaturesParam(self.mag_layer.weight, 0, "other"),
            "mag_bias": FeaturesParam(self.mag_layer.bias, 0, "bias"),
        }


class GTMulti(cl.Module):
    """
    Multi-gate variant (designed for multiple gate pre-activations, but currently single).

    Note: The multi-gate iteration in the original was vestigial - this preserves
    the interface but clarifies it's single-gate in practice.
    """

    def __init__(
        self,
        cfg: GateConfig,
        init: Initializer,
        d_in_override: int | None = None,
        targeting: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.mag_weights:
            self.mag = init._encoder.new_bias()
            self.mag.data -= 1

        self.gate_encoder = init.encoder

        self.targeting = L0Targeting(
            init.l0_target,
            scale=cfg.leniency,
            increment=0.0001 if cfg.leniency_targeting else 0.0,
        )

        self._grad_window = GRAD_WINDOWS[cfg.window_fn]

    def forward(self, x: Tensor, *, cache: cl.Cache, **kwargs) -> Tensor:
        cache.leniency = self.targeting.value

        if self._should_warmup_leniency(cache):
            self.targeting.value = cache._ancestor.trainstep / 5000

        batch_size = x.shape[0]
        mag = self.mag.unsqueeze(0).expand(batch_size, -1)

        out = apply_gating(
            mag,
            self.gate_encoder(x),
            grad_window=self._grad_window,
            d_data=x.shape[-1],
            leniency=self.targeting.value,
            training=self.training,
            uniform_noise=self.cfg.uniform_noise,
            noise_mult=self.cfg.noise_mult,
            exp_mag=True,
        )

        cache(self).targeting(out)
        return out

    def _should_warmup_leniency(self, cache: cl.Cache) -> bool:
        return (
            not self.cfg.leniency_targeting
            and cache._ancestor.has.trainstep
            and cache._ancestor.trainstep <= 5000
        )

    @property
    def features(self) -> dict[str, FeaturesParam]:
        return {"mag": FeaturesParam(self.mag, 0, "bias")}


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

Config = GateConfig  # Original name


def GT2(grad_window: GradWindowFn = sigmoid_grad_window):
    """
    Factory for backwards compatibility with original API.

    Returns a callable that applies GatedSTE with the specified gradient window.
    New code should use apply_gating() or GatedSTE.apply() directly.
    """

    def apply_fn(gate_pre, gate_post, noise, mag, leniency, d_data=768):
        # Note: gate_post was unused in original - kept for signature compat
        return GatedSTE.apply(gate_pre, noise, mag, leniency, d_data, grad_window)

    return apply_fn


import saeco
import saeco.components as co
import saeco.components.hooks.feature_hooks
from saeco.architecture import loss_prop, model_prop
from saeco.components import EMAFreqTracker, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.misc import useif


# useif
class TGArch(Architecture[GateConfig]):
    def setup(self):
        self.init._decoder.add_wrapper(
            saeco.components.hooks.feature_hooks.NormFeatures
        )
        self.init._decoder.add_wrapper(
            saeco.components.hooks.feature_hooks.OrthogonalizeFeatureGrads
        )

    @model_prop
    def model(self):
        init = self.init
        s = SAE(
            encoder=Seq(
                **useif(self.cfg.pre_bias, pre_bias=init._decoder.sub_bias),
                lin=GTTest(self.cfg, init)
                if self.cfg.mag_weights
                else BinaryEncoder(self.cfg, init),
            ),
            freqs=EMAFreqTracker(),
            # metrics=co.metrics.ActMetrics(),
            penalty=co.LinearDecayL1Penalty(40_000)
            if self.cfg.decay_l1
            else co.L1Penalty(),
            decoder=init.decoder,
        )
        s.train()
        return s

    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
