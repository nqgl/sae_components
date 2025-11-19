import torch
import torch.nn as nn
from saeco.architectures.threshgate_gradjust.other_lin import OtherLinear
from saeco.architectures.threshgate_gradjust.threshgrad import BinaryEncoder, GTTest

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.components import EMAFreqTracker, L2Loss, SparsityPenaltyLoss
from saeco.core import Seq
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig


def mlp_layer(
    d_in,
    d_hidden,
    d_out=None,
    nonlinearity=nn.LeakyReLU,
    normalize=False,
    no_resample=False,
):
    d_out = d_out or d_in
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_hidden).cuda()
    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()
    proj_in = OtherLinear(nn.Linear(d_in, d_hidden), weight_param_index=1)
    proj_in.features["bias"].resampled = False
    proj_out = OtherLinear(nn.Linear(d_hidden, d_out), weight_param_index=1)
    proj_out.features["weight"].resampled = False
    if no_resample:
        proj_out.features["bias"].resampled = False
        proj_in.features["weight"].resampled = False

    if normalize:
        proj_in = ft.NormFeatures(proj_in, index="weight", ord=2, max_only=True)

        proj_out = ft.NormFeatures(proj_out, index="weight", ord=2, max_only=True)

    return Seq(proj_in=proj_in, nonlinearity=nonlinearity, proj_out=proj_out)


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


class L1Checkpoint(cl.PassThroughModule):
    def process_data(self, x, *, cache, **kwargs):
        cache.elementwise_l1 = ...
        cache.elementwise_l1 = x.relu().mean(dim=1, keepdim=True)


nonlinearities = dict(
    relu=nn.ReLU,
    leakyrelu=nn.LeakyReLU,
    prelu=nn.PReLU,
    gelu=nn.GELU,
)


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


def deep_tg_grad_sae(
    init: Initializer,
    cfg: DeepConfig,
):
    penalty = (
        co.LinearDecayL1Penalty(cfg.decay_l1_end, end_scale=cfg.decay_l1_to)
        if cfg.decay_l1_to != 1
        else co.L1Penalty()
    )

    def dropout_no_scale(p):
        drop = torch.nn.Dropout(p=p)

        @co.ops.Lambda
        def scaleif(x):
            if drop.training:
                return x * (1 - p)
            return x

        return Seq(drop=drop, scale=scaleif)

    # lin = OtherLinear(init.d_data, init.d_data)
    if cfg.squeeze_channels > 1:
        init._decoder.d_in = init.d_dict // cfg.squeeze_channels
        init._decoder._weight_tie = None

    def dec_layer():
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

    dec_layers = [dec_layer() for _ in range(cfg.deep_dec)]
    deep = Seq(
        **useif(
            not cfg.norm_deep_dec,
            l1_checkpoint=L1Checkpoint(),
        ),
        **useif(
            cfg.resid,
            resid=cl.collections.seq.ResidualSeq(*dec_layers),
        ),
        **useif(not cfg.resid, layers=Seq(*dec_layers)),
        **useif(
            not cfg.norm_deep_dec,
            l1_restore=ConservedL1(max_only=cfg.l1_max_only),
        ),
    )

    model = Seq(
        encoder=Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            **useif(
                cfg.deep_enc,
                resids=cl.collections.seq.ResidualSeq(
                    layer=Seq(
                        mlp_layer(
                            init.d_data,
                            init.d_data * 4,
                            nonlinearity=nn.PReLU,
                            no_resample=True,
                            # scale=0.1,
                        ),
                        nn.LayerNorm(init.d_data),
                    ),
                ),
            ),
            lin=(
                gt := (
                    GTTest(cfg, init)
                    if cfg.mag_weights
                    else BinaryEncoder(
                        cfg,
                        init,
                        apply_targeting_externally=(cfg.squeeze_channels > 1),
                        signed_mag=cfg.signed_mag,
                        penalty=(
                            Seq(
                                penalty=penalty,
                                metrics=co.metrics.ActMetrics(),
                            )
                            if cfg.measure_in_gate
                            else None
                        ),
                    )
                )
            ),
        ),
        freqs=EMAFreqTracker(),
        **useif(
            cfg.squeeze_channels > 1,
            pre_squeeze_metrics=co.metrics.ActMetrics("pre-squeeze"),
            squeeze=Seq(
                split=co.ops.Lambda(
                    lambda x: torch.split(x, init.d_dict // cfg.squeeze_channels, dim=1)
                ),
                route_add=cl.collections.Router(
                    *[cl.ops.Identity() for _ in range(cfg.squeeze_channels)]
                ).reduce(lambda a, b: a + b, binary=True),
            ),
        ),
        **useif(
            not cfg.measure_in_gate,
            metrics=co.metrics.ActMetrics(),
        ),
        **useif(cfg.squeeze_channels > 1, gt_targeting=gt.targeting),
        **useif(cfg.dropout > 0, dropout=dropout_no_scale(cfg.dropout)),
        **useif(
            (not cfg.measure_in_gate) and (not cfg.penalize_after),
            penalty=penalty,
        ),
        **useif(cfg.deep_dec, deep=deep),
        **useif((not cfg.measure_in_gate) and cfg.penalize_after, penalty=penalty),
        deep_metrics=co.metrics.ActMetrics("deep_metrics"),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder.set_resampled(cfg.resample_dec),
            ),
        ).set_resampled(cfg.squeeze_channels == 1),
    )

    models = [model]
    losses = dict(
        L2_loss=L2Loss(model),
        sparsity_loss=SparsityPenaltyLoss(model),
    )
    return models, losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=deep_tg_grad_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    pass
