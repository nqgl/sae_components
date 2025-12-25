import torch
import torch.nn as nn
from saeco.architectures.sweep_tg.mlp_layer import mlp_layer
from saeco.architectures.sweep_tg.sthreshgrad import BinaryEncoder, GTTest

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import EMAFreqTracker, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig, do_sweep
from saeco.trainer.runner import TrainingRunner


# ================================
# Configuration
# ================================
class DeepConfig(SweepableConfig):
    pre_bias: bool = False
    uniform_noise: bool = True
    noise_mult: float = 0.1
    exp_mag: bool = True
    mag_weights: bool = True
    window_fn: str = "sig"
    decay_l1_to: float = 1
    decay_l1_end: int = 40_000
    leniency_targeting: bool = False
    leniency: float = 1
    deep_enc: bool = False
    deep_dec: int = True
    l1_max_only: bool = False
    use_layernorm: bool = False
    penalize_after: bool = False
    resid: bool = True
    dec_mlp_expansion_factor: int | float = 8
    resample_dec: bool = True
    dec_mlp_nonlinearity: str = "relu"
    norm_deep_dec: bool = False
    squeeze_channels: int = 1
    dropout: float = 0.0
    signed_mag: bool = False
    measure_in_gate: bool = True


# ================================
# Custom Modules
# ================================
class L1Checkpoint(cl.PassThroughModule):
    def process_data(self, x, *, cache, **kwargs):
        # Captures elementwise L1 norm for later use
        cache.elementwise_l1 = x.relu().mean(dim=1, keepdim=True)


class ConservedL1(cl.Module):
    def __init__(self, max_only) -> None:
        super().__init__()
        self.max_only = max_only

    def forward(self, x: torch.Tensor, *, cache: cl.Cache, **kwargs):
        l1_caches = cache._parent.search("elementwise_l1")
        assert len(l1_caches) == 1
        l1 = l1_caches[0].elementwise_l1
        x_l1 = x.abs().mean(dim=1, keepdim=True)
        scale: torch.Tensor = l1 / (x_l1 + 1e-7)
        if self.max_only:
            scale = scale.clamp(max=1)
        return x * scale


# ================================
# Nonlinearities Dictionary
# ================================
nonlinearities = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "gelu": nn.GELU,
}


# ================================
# Helper Functions for Model Components
# ================================
def make_penalty(cfg: DeepConfig):
    """Create the L1 penalty component, possibly with linear decay."""
    if cfg.decay_l1_to != 1:
        return co.LinearDecayL1Penalty(cfg.decay_l1_end, end_scale=cfg.decay_l1_to)
    else:
        return co.L1Penalty()


def make_dropout(cfg: DeepConfig):
    """Create a dropout with no scaling on the forward pass."""
    p = cfg.dropout
    drop = torch.nn.Dropout(p=p)

    @co.ops.Lambda
    def scaleif(x):
        # Scale output by (1-p) only during training, to keep activation magnitude stable.
        if drop.training:
            return x * (1 - p)
        return x

    return Seq(drop=drop, scale=scaleif) if p > 0 else None


def dec_layer(init: Initializer, cfg: DeepConfig):
    """Create a single decoder layer."""
    return Seq(
        mlp=mlp_layer(
            init.d_dict // cfg.squeeze_channels,
            init.d_data * cfg.dec_mlp_expansion_factor,
            nonlinearity=nonlinearities[cfg.dec_mlp_nonlinearity],
            normalize=cfg.norm_deep_dec,
            no_resample=cfg.squeeze_channels > 1,
        ),
        **useif(
            cfg.use_layernorm,
            ln=nn.LayerNorm(init.d_dict // cfg.squeeze_channels),
        ),
    )


def build_decoder(init: Initializer, cfg: DeepConfig):
    """Build the deep decoder portion of the model."""
    dec_layers = [dec_layer(init, cfg) for _ in range(cfg.deep_dec)]

    # Construct the deep portion (residual or sequential)
    deep_part = Seq(
        **useif(not cfg.norm_deep_dec, l1_checkpoint=L1Checkpoint()),
        **useif(cfg.resid, resid=cl.collections.seq.ResidualSeq(*dec_layers)),
        **useif(not cfg.resid, layers=Seq(*dec_layers)),
        **useif(
            not cfg.norm_deep_dec, l1_restore=ConservedL1(max_only=cfg.l1_max_only)
        ),
    )

    return deep_part


def build_encoder(init: Initializer, cfg: DeepConfig, penalty):
    """Build the encoder (including gating logic)."""
    gating_enc = (
        GTTest(cfg, init)
        if cfg.mag_weights
        else BinaryEncoder(
            cfg,
            init,
            apply_targeting_externally=(cfg.squeeze_channels > 1),
            signed_mag=cfg.signed_mag,
            penalty=(
                Seq(penalty=penalty, metrics=co.metrics.ActMetrics())
                if cfg.measure_in_gate
                else None
            ),
        )
    )

    encoder_layers = []
    if cfg.pre_bias:
        encoder_layers.append(("pre_bias", init._decoder.sub_bias()))

    if cfg.deep_enc:
        # Example deep encoder block
        encoder_layers.append(
            (
                "resids",
                cl.collections.seq.ResidualSeq(
                    layer=Seq(
                        mlp_layer(
                            init.d_data,
                            init.d_data * 4,
                            nonlinearity=nn.PReLU,
                            no_resample=True,
                        ),
                        nn.LayerNorm(init.d_data),
                    ),
                ),
            )
        )

    encoder_layers.append(("lin", gating_enc))

    return gating_enc, Seq(*[Seq(**{name: layer}) for name, layer in encoder_layers])


def build_squeeze(cfg: DeepConfig, init: Initializer):
    """Build the squeezing stage if needed."""
    if cfg.squeeze_channels > 1:
        return Seq(
            pre_squeeze_metrics=co.metrics.ActMetrics("pre-squeeze"),
            squeeze=Seq(
                split=co.ops.Lambda(
                    lambda x: torch.split(x, init.d_dict // cfg.squeeze_channels, dim=1)
                ),
                route_add=cl.collections.Router(
                    *[cl.ops.Identity() for _ in range(cfg.squeeze_channels)]
                ).reduce(lambda a, b: a + b, binary=True),
            ),
        )
    return None


def deep_tg_grad_sae(init: Initializer, cfg: DeepConfig):
    """Build the full model based on the configuration."""
    penalty = make_penalty(cfg)
    gating_enc, encoder = build_encoder(init, cfg, penalty)
    deep = build_decoder(init, cfg)
    drop = make_dropout(cfg)
    squeeze_block = build_squeeze(cfg, init)

    # Adjust input dimension if we squeeze channels
    if cfg.squeeze_channels > 1:
        init._decoder.d_in = init.d_dict // cfg.squeeze_channels
        init._decoder._weight_tie = None

    decoder = ft.OrthogonalizeFeatureGrads(
        ft.NormFeatures(
            init.decoder.set_resampled(cfg.resample_dec),
        ),
    ).set_resampled(cfg.squeeze_channels == 1)

    model = Seq(
        encoder=encoder,
        freqs=EMAFreqTracker(),
        **useif(cfg.squeeze_channels > 1, squeezer=squeeze_block),
        **useif(not cfg.measure_in_gate, metrics=co.metrics.ActMetrics()),
        **useif(cfg.squeeze_channels > 1, gt_targeting=gating_enc.targeting),
        **useif(cfg.dropout > 0, dropout=drop),
        **useif(
            (not cfg.measure_in_gate) and (not cfg.penalize_after), penalty=penalty
        ),
        **useif(cfg.deep_dec, deep=deep),
        **useif((not cfg.measure_in_gate) and cfg.penalize_after, penalty=penalty),
        deep_metrics=co.metrics.ActMetrics("deep_metrics"),
        decoder=decoder,
    )

    losses = {
        "L2_loss": L2Loss(model),
        "sparsity_loss": SparsityPenaltyLoss(model),
    }

    return [model], losses


# ================================
# Training/Running
# ================================
def run(cfg: DeepConfig):
    tr = TrainingRunner(cfg, model_fn=deep_tg_grad_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    # These imports could be done in a separate config file.
    pass
