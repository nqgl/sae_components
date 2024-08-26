import torch

import torch.nn as nn

from saeco.architectures.threshgate_gradjust.threshgate import (
    BinaryEncoder,
    GTTest,
    OtherLinear,
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
    mlp_layer,
)


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
    deep_dec: bool = True
    l1_max_only: bool = False
    use_layernorm: bool = False


class L1Checkpoint(cl.PassThroughModule):
    def process_data(self, x, *, cache, **kwargs):
        cache.elementwise_l1 = ...
        cache.elementwise_l1 = x.abs().mean(dim=1, keepdim=True)


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

    # lin = OtherLinear(init.d_data, init.d_data)
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
                            scale=0.1,
                        ),
                        nn.LayerNorm(init.d_data),
                    ),
                ),
            ),
            lin=GTTest(cfg, init) if cfg.mag_weights else BinaryEncoder(cfg, init),
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        penalty=(
            co.LinearDecayL1Penalty(40_000, end_scale=cfg.decay_l1_to)
            if cfg.decay_l1_to != 1
            else co.L1Penalty()
        ),
        **useif(
            cfg.deep_dec,
            deep=Seq(
                l1_checkpoint=L1Checkpoint(),
                resid=cl.collections.seq.ResidualSeq(
                    layer=Seq(
                        mlp=mlp_layer(
                            init.d_dict,
                            init.d_dict // 2,
                            nonlinearity=nn.LeakyReLU(negative_slope=3e-3),
                            scale=0.1,
                        ),
                        **useif(
                            cfg.use_layernorm,
                            ln=nn.LayerNorm(init.d_dict),
                        ),
                    ),
                ),
                l1_restore=ConservedL1(max_only=cfg.l1_max_only),
            ),
        ),
        decoder=ft.OrthogonalizeFeatureGrads(
            ft.NormFeatures(
                init.decoder,
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
    if not cfg.arch_cfg.leniency_targeting:
        cfg.train_cfg.l0_targeting_enabled = True
        assert cfg.arch_cfg.decay_l1_to == 1
        # cfg.arch_cfg.leniency = 1
        cfg.train_cfg.coeffs["sparsity_loss"] = 1e-4
    assert cfg.arch_cfg.deep_enc or cfg.arch_cfg.deep_dec
    tr = TrainingRunner(cfg, model_fn=deep_tg_grad_sae)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
else:
    from .tg_grad_deep_config import cfg, PROJECT
