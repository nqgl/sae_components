from enum import IntEnum

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

from torch.autograd.function import Function


class AttributedPositiveSum(Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim):
        out = x.sum(dim=dim, keepdim=keepdim)
        ctx.save_for_backward(x, out)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            u_out = out.unsqueeze(ctx.dim)
        proportions = x / u_out
        grad_input = grad_output * proportions
        return grad_input, None, None


def attributed_positive_sum(x, dim, keepdim=False):
    return AttributedPositiveSum.apply(x, dim, keepdim)


class ReuseLayer(IntEnum):
    NONE = 0
    ACC_ACTS = 1
    ACC_PRE = 2
    ACC_ACTS_L1_SHIELD = 3
    ACC_PRE_L1_SHIELD = 4
    ACC_ACTS_L1_SHARE = 5
    ACC_PRE_L1_SHARE = 6


class ResidConfig(SweepableConfig):
    pre_bias: bool = False
    layers: int = 4
    individual_dec_bias: bool = False
    individual_enc_bias: bool = False
    anth_scale: bool = False
    thresh_range: tuple[float, float] = (0, 0)
    reuse_layer: ReuseLayer = False

    @property
    def l1_shielding(self):
        return self.reuse_layer in (
            ReuseLayer.ACC_ACTS_L1_SHIELD,
            ReuseLayer.ACC_PRE_L1_SHIELD,
        )

    @property
    def l1_sharing(self):
        return self.reuse_layer in (
            ReuseLayer.ACC_ACTS_L1_SHARE,
            ReuseLayer.ACC_PRE_L1_SHARE,
        )


GLOBD = 0


class ResidModule(cl.Module):
    def __init__(self, *modules, cfg: ResidConfig, acts_module, init: Initializer):
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

        for i, (enc, dec) in enumerate(self.contained):
            thresh = self.cfg.thresh_range[0] - (
                self.cfg.thresh_range[0] - self.cfg.thresh_range[1]
            ) * i / (len(self.contained) - 1)
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


class ReuseResidModule(cl.Module):
    def __init__(self, *modules, cfg: ResidConfig, acts_module, init: Initializer):
        super().__init__()
        self.contained = nn.ModuleList(modules)
        self.metrics = acts_module
        self.preact_metrics = co.metrics.PreActMetrics()
        self.cfg = cfg
        self.accumulate_pre = self.cfg.reuse_layer in (
            ReuseLayer.ACC_PRE,
            ReuseLayer.ACC_PRE_L1_SHIELD,
            ReuseLayer.ACC_PRE_L1_SHARE,
        )
        self.d_dict = init.d_dict

    def forward(self, x, *, cache: cl.Cache):
        acc_pred = 0
        pre_acts = 0
        acts = 0
        global GLOBD
        if GLOBD % 100 == 50:
            print("begin")
            print((x).pow(2).mean())
        GLOBD += 1
        if self.cfg.l1_sharing or self.cfg.l1_shielding:
            activity_mask = torch.zeros(
                (self.cfg.layers, x.shape[0], self.d_dict),
                dtype=torch.bool,
                device=x.device,
            )
            activities = torch.zeros(
                (self.cfg.layers, x.shape[0], self.d_dict),
                dtype=torch.float32,
                device=x.device,
            )
            l1_acts = 0
        else:
            activity_mask = None
            l1_acts = None
        for i, (enc, dec) in enumerate(self.contained):
            thresh = self.cfg.thresh_range[0] - (
                self.cfg.thresh_range[0] - self.cfg.thresh_range[1]
            ) * i / (len(self.contained) - 1)
            layer_pre_acts = enc(x - acc_pred)
            pre_acts += layer_pre_acts
            if self.accumulate_pre:
                layer_acts = torch.where(pre_acts > thresh, pre_acts, 0)
                acts = layer_acts
            else:
                layer_acts = torch.where(layer_pre_acts > thresh, layer_pre_acts, 0)
                acts = acts + layer_acts
            if self.cfg.l1_shielding:
                l1_acts = l1_acts + torch.where(
                    activity_mask.any(dim=0), layer_acts.detach(), layer_acts
                )

            if activity_mask is not None:
                activity_mask[i] = layer_acts > 0
                activities[i] = layer_acts
            pred = dec(acts)
            acc_pred = pred

            # acc_pred += pred  # TODO
            if GLOBD % 100 == 50:
                print(
                    f"{layer_acts.count_nonzero(dim=1).float().mean().item():.2f}, {(x - acc_pred).pow(2).mean()}"
                )
        if self.cfg.l1_sharing:
            active_counts = activity_mask.sum(dim=0)
            l1_acts = attributed_positive_sum(activities, dim=0)
            # l1_acts = torch.where(
            #     active_counts > 0,
            #     acts / active_counts
            #     + acts.detach() * (active_counts - 1) / active_counts,
            #     0,
            # )
        cache(self).metrics(acts if l1_acts is None else l1_acts)
        cache(self).preact_metrics(pre_acts)
        return acc_pred


class SlicedEncoder(nn.Module):
    def __init__(self, lin: LinEncoder, split: slice, bias=None):
        super().__init__()
        self.lin = lin
        self.split = split
        self.bias = bias

    def forward(self, x):
        return torch.nn.functional.linear(
            x,
            self.lin.get_weight()[self.split],
            self.lin.get_bias()[self.split] if self.bias is None else self.bias,
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
    encoder: LinEncoder,
    decoder: LinDecoder,
    d_dict,
    cfg: ResidConfig,
    init: Initializer,
):
    if cfg.reuse_layer:
        split = slice(0, d_dict)
        return [
            nn.ModuleList(
                [SlicedEncoder(encoder, split), SlicedDecoder(decoder, split, init)]
            )
            for _ in range(cfg.layers)
        ]
    nsplit = cfg.layers
    assert d_dict % nsplit == 0
    split_size = d_dict // nsplit
    splits = [slice(i, i + split_size) for i in range(0, d_dict, split_size)]
    return [
        nn.ModuleList(
            [
                SlicedEncoder(
                    encoder,
                    split,
                    bias=(
                        None
                        if not cfg.individual_enc_bias
                        else init._encoder.new_bias()
                    ),
                ),
                SlicedDecoder(decoder, split, init),
            ]
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
        encoder=(ReuseResidModule if cfg.reuse_layer else ResidModule)(
            *split_pairs(
                encoder=encoder,
                decoder=decoder,
                d_dict=init.d_dict,
                cfg=cfg,
                init=init if cfg.individual_dec_bias else None,
            ),
            cfg=cfg,
            acts_module=Metrics(
                l1=penalizer,
                freqs=EMAFreqTracker(),
                acts=co.metrics.ActMetrics(),
            ),
            init=init,
        ),
        **useif(not cfg.individual_dec_bias, bias=cl.ops.Add(init._decoder.new_bias())),
    )
    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses
