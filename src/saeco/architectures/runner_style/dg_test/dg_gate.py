import torch

import torch.nn as nn

import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
from saeco.core import ReuseForward
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
import einops

from saeco.sweeps.sweepable_config.Swept import Swept


class Config(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True
    # decay_l1_penalty: float =

    relu_first_acts: bool = Swept(True, False)
    gelu_mid: bool = Swept(True, False)

    detach_enc: bool = False
    segment_n: int = 1
    squeeze_dg: None | int = None


class SemgentedDG(nn.Module):
    def __init__(self, d_dict, segmentation_n, bias=True):
        super().__init__()
        assert d_dict % segmentation_n == 0
        seg_size = d_dict // segmentation_n
        dg = torch.eye(seg_size).unsqueeze(0).repeat(segmentation_n, 1, 1)
        self.dg = nn.Parameter(dg)
        self.segmentation_n = segmentation_n
        self.d_dict = d_dict
        self.seg_size = seg_size
        self.bias = nn.Parameter(torch.zeros(d_dict)) if bias else 0

    def forward(self, x):
        x = x.view(x.shape[0], self.segmentation_n, self.seg_size)
        x = torch.einsum("bns,nsd->bnd", x, self.dg)
        return einops.rearrange(x, "b n d -> b (n d)") + self.bias


def gated_dg_sae(
    init: Initializer,
    cfg: Config,
):
    if cfg.segment_n == 1:
        dg = nn.Linear(init.d_dict, init.d_dict, bias=True)
        dg.weight.data = torch.eye(init.d_dict)
    elif cfg.squeeze_dg is not None:
        dg = Seq(
            nn.Linear(init.d_dict, cfg.squeeze_dg, bias=True),
            nn.PReLU(),
            nn.Linear(cfg.squeeze_dg, init.d_dict, bias=True),
        )
    else:
        dg = SemgentedDG(init.d_dict, 4, bias=True)
    init._encoder.bias = False
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    enc_mag = Seq(
        **useif(
            cfg.pre_bias,
            pre_bias=ReuseForward(init._decoder.sub_bias()),
        ),
        r_mag=cl.ops.MulParallel(
            identity=ReuseForward(init.encoder),
            exp_r=co.Lambda(
                func=lambda x: torch.exp(x),
                module=init.dict_bias(),
            ),
        ),
        bias=init.new_encoder_bias(),
        nonlinearity=nn.ReLU(),
    )

    enc_gate = ReuseForward(
        Seq(
            **useif(
                cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(left=cl.ops.Identity(), right=init.decoder.bias).reduce(
                        (lambda l, r: l - r.detach())
                    )
                    if cfg.detach
                    else ReuseForward(init._decoder.sub_bias())
                ),
            ),
            weight=ReuseForward(init.encoder),
            **useif(cfg.detach_enc, detach=co.Lambda(lambda x: x.detach())),
            bias=init.new_encoder_bias(),
            dg=cl.Parallel(
                Seq(nn.GELU() if cfg.gelu_mid else nn.ReLU(), dg),
                nn.ReLU() if cfg.relu_first_acts else cl.ops.Identity(),
            ).reduce(lambda a, b: (a + b) / 2),
            relu=nn.ReLU(),
        )
    )

    # models
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag,
            gate=co.ops.Thresh(enc_gate),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=init.decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate,  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=init._decoder.detached if cfg.detach else init.decoder,
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses


from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=gated_dg_sae)
    for i in range(5):
        tr.trainer.train(num_steps=10_000)
        tr.trainer.save()
        tr.trainer.cfg.batch_size *= 2


if __name__ == "__main__":
    do_sweep(True)
else:
    from .dg_gate_config import cfg, PROJECT
