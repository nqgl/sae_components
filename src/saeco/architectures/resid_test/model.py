import saeco.components as co
import saeco.components.features.features as ft
import saeco.core as cl
import torch

import torch.nn as nn
from saeco.components import EMAFreqTracker, L1Penalty, L2Loss, SparsityPenaltyLoss
from saeco.components.features.linear_type import LinDecoder, LinEncoder

from saeco.components.hooks.clipgrad import ClipGrad
from saeco.components.metrics.metrics import Metrics
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.core import Seq

from saeco.initializer import Initializer

from saeco.misc import useif
from saeco.sweeps import do_sweep, SweepableConfig
from saeco.trainer.runner import TrainingRunner


class ResidConfig(SweepableConfig):
    pre_bias: bool = False
    layers: int = 4
    individual_dec_bias: bool = False
    anth_scale: bool = False
    thresh_range: tuple[float, float] = (0, 0)


GLOBD = 0


class ResidModule(cl.Module):
    def __init__(self, *modules, cfg, acts_module):
        super().__init__()
        self.contained = nn.ModuleList(modules)
        self.metrics = acts_module
        self.preact_metrics = co.metrics.PreActMetrics()
        self.cfg = cfg

    def forward(self, x, *, cache: cl.Cache):
        acc_pred = 0
        acts_l = []
        pre_acts_l = []
        global GLOBD
        if GLOBD % 100 == 50:
            print("begin")
            print((x).pow(2).mean())
        GLOBD += 1
        thresh_range = (1, 0)
        for i, (enc, dec) in enumerate(self.contained):
            thresh = thresh_range[0] - (thresh_range[0] - thresh_range[1]) * i / len(
                self.contained
            )
            pre_acts = enc(x - acc_pred)
            acts = torch.where(pre_acts > thresh, pre_acts, 0)
            pre_acts_l.append(pre_acts)
            acts_l.append(acts)
            pred = dec(acts)
            acc_pred += pred
            if GLOBD % 100 == 50:
                print(
                    f"{acts.count_nonzero(dim=1).float().mean().item():.2f}, {(x - acc_pred).pow(2).mean()}"
                )
        cache(self).metrics(torch.cat(acts_l, dim=1))
        cache(self).preact_metrics(torch.cat(pre_acts_l, dim=1))
        return acc_pred


class SlicedEncoder(nn.Module):
    def __init__(self, lin: LinEncoder, split: slice):
        super().__init__()
        self.lin = lin
        self.split = split

    def forward(self, x):
        return torch.nn.functional.linear(
            x, self.lin.get_weight()[self.split], self.lin.get_bias()[self.split]
        )


class SlicedDecoder(nn.Module):
    def __init__(self, lin: LinDecoder, split: slice, init: Initializer):
        super().__init__()
        self.lin = lin
        self.split = split
        self.bias = None if init is None else init._decoder.new_bias()

    def forward(self, x):
        return torch.nn.functional.linear(
            x, self.lin.get_weight()[:, self.split], self.bias
        )


def split_pairs(
    encoder: LinEncoder, decoder: LinDecoder, d_dict, nsplit, init: Initializer
):
    assert d_dict % nsplit == 0
    split_size = d_dict // nsplit
    splits = [slice(i, i + split_size) for i in range(0, d_dict, split_size)]
    return [
        nn.ModuleList(
            [SlicedEncoder(encoder, split), SlicedDecoder(decoder, split, init)]
        )
        for split in splits
    ]


def resid_sae(
    init: Initializer,
    cfg: ResidConfig,
):
    init._decoder.const_init_bias(0)
    init._encoder.const_init_bias(0)
    encoder = init.encoder.resampled()
    penalizer = L1PenaltyScaledByDecoderNorm() if cfg.anth_scale else L1Penalty()
    decoder = (
        penalizer.set_decoder(init.decoder.resampled())
        if cfg.anth_scale
        else ft.OrthogonalizeFeatureGrads(ft.NormFeatures(init.decoder.resampled()))
    )
    model_full = Seq(
        **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        encoder=ResidModule(
            *split_pairs(
                encoder=encoder,
                decoder=decoder,
                d_dict=init.d_dict,
                nsplit=cfg.layers,
                init=init if cfg.individual_dec_bias else None,
            ),
            cfg=cfg,
            acts_module=Metrics(
                l1=penalizer,
                freqs=EMAFreqTracker(),
                acts=co.metrics.ActMetrics(),
            ),
        ),
        **useif(not cfg.individual_dec_bias, bias=cl.ops.Add(init._decoder.new_bias())),
    )
    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses
