import torch

import torch.nn as nn

from saeco.architectures.threshgate_gradjust.other_lin import OtherLinear
from saeco.architectures.threshgate_gradjust.threshgate import (
    BinaryEncoder,
    GTTest,
)
import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl

from saeco.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.core import Seq

from saeco.misc import useif
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.sweeps import SweepableConfig
from saeco.initializer.tools import (
    reused,
    weight,
    bias,
)


def mlp_layer(d_in, d_hidden, d_out=None, nonlinearity=nn.LeakyReLU, normalize=False):
    d_out = d_out or d_in
    if nonlinearity is nn.PReLU:
        nonlinearity = nn.PReLU(d_hidden).cuda()
    if isinstance(nonlinearity, type):
        nonlinearity = nonlinearity()
    proj_in = OtherLinear(nn.Linear(d_in, d_hidden), weight_param_index=1)
    proj_in.features["bias"].resampled = False
    proj_out = OtherLinear(nn.Linear(d_hidden, d_out), weight_param_index=1)
    proj_out.features["weight"].resampled = False

    if normalize:
        proj_in = ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(proj_in, index="weight", ord=2), index="weight"
        )
        proj_out = ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(proj_out, index="weight", ord=2), index="weight"
        )
    return Seq(proj_in=proj_in, nonlinearity=nonlinearity, proj_out=proj_out)


class DeepConfig(SweepableConfig):
    pre_bias: bool = False
    uniform_noise: bool = True
    noise_mult: float = 0.1
    exp_mag: bool = True
    mag_weights: bool = True
    window_fn: str = "sig"
    decay_l1_to: float = 1
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
    penalty = dict(
        penalty=(
            co.LinearDecayL1Penalty(40_000, end_scale=cfg.decay_l1_to)
            if cfg.decay_l1_to != 1
            else co.L1Penalty()
        )
    )
    # lin = OtherLinear(init.d_data, init.d_data)

    def dec_layer():
        return Seq(
            mlp=mlp_layer(
                init.d_dict,
                init.d_data * cfg.dec_mlp_expansion_factor,
                nonlinearity=nonlinearities[cfg.dec_mlp_nonlinearity],
                normalize=cfg.norm_deep_dec,
            ),
            **useif(
                cfg.use_layernorm,
                ln=nn.LayerNorm(init.d_dict),
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
            # **useif(
            #     cfg.deep_enc,
            #     resids=cl.collections.seq.ResidualSeq(
            #         layer=Seq(
            #             mlp_layer(
            #                 init.d_data,
            #                 init.d_data * 4,
            #                 nonlinearity=nn.PReLU,
            #                 scale=0.1,
            #             ),
            #             nn.LayerNorm(init.d_data),
            #         ),
            #     ),
            # ),
            lin=GTTest(cfg, init) if cfg.mag_weights else BinaryEncoder(cfg, init),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        **useif(not cfg.penalize_after, **penalty),
        **useif(cfg.deep_dec, deep=deep),
        **useif(cfg.penalize_after, **penalty),
        deep_metrics=co.metrics.ActMetrics("deep_metrics"),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder.set_resampled(cfg.resample_dec),
            ),
        ),
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
    tr.trainer.save()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .tg_grad_deep_config import cfg, PROJECT
