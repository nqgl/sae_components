from functools import cached_property

import einops
import nnsight
import saeco.components.features.features as ft

import saeco.core as cl
import torch
import torch.nn as nn

import torch.nn.functional as F

from saeco.architecture import Architecture, aux_model_prop, loss_prop, model_prop, SAE

from saeco.components import L2Loss, Lambda, Loss, SparsityPenaltyLoss
from saeco.components.features.features_param import FeaturesParam
from saeco.components.ops import Indexer
from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward

from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.initializer.initializer import Initializer
from saeco.misc import useif
from saeco.misc.nnsite import getsite, setsite
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from saeco.trainer.evaluation_protocol import ReconstructionEvaluatorFunctionProtocol
from saeco.trainer.recons import to_losses
from saeco.trainer.trainable import Trainable


class MatryoshkaCLTConfig(SweepableConfig):
    n_sites: int = 12
    n_nestings: int = 3


def with_sae_runner(
    model: nnsight.LanguageModel, encoder: Trainable, cfg: ActsDataConfig
):
    def saerunner(tokens):
        mlp_inputs = []

        with model.trace(tokens):
            for i in range(12):
                mlp_input_site = f"transformer.h.{i}.input"
                mlp_output_site = f"transformer.h.{i}.output"

                mlp_input = getsite(model, mlp_input_site).save()

                mlp_inputs.append(mlp_input)

                acts_re = nnsight.apply(
                    lambda x, layer_idx=i: encoder.raw_model.reconstruct_layer(
                        torch.cat(x, dim=-1).float()[0], layer_idx
                    ).to(x[0].dtype),
                    mlp_inputs,
                )

                setsite(model, mlp_output_site, acts_re)

            out = model.output.logits.save()

        return out

    return saerunner


def zero_ablated_runner(model: nnsight.LanguageModel, cfg: ActsDataConfig):
    def zrunner(tokens):
        with model.trace(tokens) as tracer:
            for site in cfg.sites:
                if "input" in site:
                    lm_acts = getsite(model, site)
                    acts_re = nnsight.apply(torch.zeros_like, lm_acts)
                    patch_in = acts_re
                    setsite(model, site, patch_in)

            out = model.output.logits.save()
        return out

    return zrunner


def normal_runner(model: nnsight.LanguageModel, cfg: ActsDataConfig):
    def nrunner(tokens):
        with model.trace(tokens):
            out = model.output.logits.save()
        return out

    return nrunner


@torch.inference_mode()
def get_multisite_recons_loss(
    llm,
    sae: Trainable,
    tokens=None,
    num_batches=10,
    cfg: ActsDataConfig = None,
    batch_size=1,
    cast_fn=...,
):
    cfg = cfg or sae.cfg
    loss_list = []

    with_sae = to_losses(with_sae_runner(llm, sae, cfg))
    zero = to_losses(zero_ablated_runner(llm, cfg))
    normal = to_losses(normal_runner(llm, cfg))
    rand_tokens = tokens[torch.randperm(len(tokens))]
    with cast_fn():
        for i in range(num_batches):
            batch_tokens = rand_tokens[i * batch_size : (i + 1) * batch_size].cuda()
            zeroed = zero(batch_tokens)
            re = with_sae(batch_tokens)
            loss = normal(batch_tokens)
            loss_list.append((loss, re, zeroed))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    return {
        "recons_score": score,
        "nats_lost": recons_loss - loss,
        "loss": loss,
        "recons_loss": recons_loss,
        "zero_ablation_loss": zero_abl_loss,
    }


class MatryoshkaLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y.unsqueeze(0) - y_pred) ** 2)


def multilayer_decoder(weight: torch.Tensor):
    def decode(x: torch.Tensor):
        # have to do the .to(x.dtype) bc something where the autocast doesn't work
        # didn't look into it enough to deeply understand it

        # This applies the per-layer decoder on each layer of input data then sums them all.
        return torch.einsum("lbd,ldo->bo", x, weight.to(x.dtype))

    return decode


def slice_dim(low, high, dim):
    return (slice(None),) * dim + (slice(low, high),)


def split_tensor(x, bounds, dim):
    if bounds[-1] != x.shape[dim]:
        #        bounds = [0] + bounds + [x.shape[dim]]
        bounds = [0] + bounds
    else:
        bounds = [0] + bounds

    if dim == -1:
        dim = x.dim()
    return [x[slice_dim(bounds[i], bounds[i + 1], dim)] for i in range(len(bounds) - 1)]


class SplittableDecoder(
    cl.Module, ft.OrthogonalizeFeatureGradsMixin, ft.NormFeaturesMixin
):
    def __init__(self, d_layer_dict, num_layers, d_layer_data, nesting_boundaries):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_layers * d_layer_dict, d_layer_data)
        ).cuda()
        self.d_dict = d_layer_dict
        self.n_nestings = len(nesting_boundaries)
        self.splits = sorted(nesting_boundaries)
        self.n_layers = num_layers
        self.d_data = d_layer_data

    def forward(self, acts, *, cache: SAECache):
        # acts is [n_layers, batch_dim, d_layer_dict]
        return torch.stack(
            [
                dec(x_i)
                for (dec, x_i) in zip(
                    self.split_dec(self.splits), split_tensor(acts, self.splits, dim=2)
                )
            ],
            dim=0,
        ).cumsum(dim=0)

    def split_dec(self, bounds: list[int]):
        return [
            multilayer_decoder(w)
            for w in split_tensor(
                einops.rearrange(
                    self.weight,
                    "(l d_dict) d_data -> l d_dict d_data",
                    d_dict=self.d_dict,
                ),
                bounds,
                dim=1,
            )
        ]

    @cached_property
    def features(self):
        return {
            "weight": FeaturesParam(
                self.weight,
                feature_index=0,
                feature_parameter_type=FeaturesParam.FPTYPES.dec,
            ),
        }


class AddSplitDecoderBias(cl.Module):
    def __init__(self, num_layers, d_layer_data):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(num_layers * d_layer_data)).cuda()

    def forward(self, x, *, cache: SAECache):
        return x + self.bias


class MatryoshkaCLTDecoder(cl.Module):
    def __init__(
        self,
        d_layer_data: int,
        d_layer_dict: int,
        nesting_boundaries: list[int],
        cfg: MatryoshkaCLTConfig,
    ):
        super().__init__()

        self.d_layer_data = d_layer_data
        self.d_layer_dict = d_layer_dict

        self.bias = AddSplitDecoderBias(
            num_layers=cfg.n_sites, d_layer_data=d_layer_data
        )

        self.decoders = [
            SplittableDecoder(
                d_layer_dict=d_layer_dict,
                num_layers=n + 1,
                d_layer_data=d_layer_data,
                nesting_boundaries=nesting_boundaries,
            )
            for n in range(cfg.n_sites)
        ]
        self.decode = Seq(
            route_to_decoders=cl.Parallel(
                *[Indexer.L[: i + 1] for i in range(cfg.n_sites)]
            ).reduce(lambda *x: x),
            splittable_decoders=cl.Router(*self.decoders).reduce(
                lambda *x: torch.cat(x, dim=-1)
            ),
            decoder_bias=self.bias,
        )

    def decode_single_layer(
        self, encodings: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        weight = self.decoders[layer_idx].weight

        batch_size = encodings.size(1)

        reshaped_encodings = encodings.permute(1, 0, 2).reshape(
            batch_size, self.d_layer_dict * (layer_idx + 1)
        )

        return F.linear(
            reshaped_encodings,
            weight.T,
            self.bias.bias[
                layer_idx * self.d_layer_data : (layer_idx + 1) * self.d_layer_data
            ],
        )

    def forward(self, x, *, cache: SAECache):
        return cache(self).decode(x)


class MatryoshkaCLT(Architecture[MatryoshkaCLTConfig]):
    def setup(self):
        assert self.init.d_dict % self.cfg.n_sites == 0
        assert self.init.d_data % self.cfg.n_sites == 0
        self.d_layer_dict = self.init.d_dict // self.cfg.n_sites
        self.d_layer_data = self.init.d_data // self.cfg.n_sites

        assert (2 ** (self.cfg.n_nestings - 1)) <= self.d_layer_dict

        self.nesting_boundaries = self.boundary_generator()

    def boundary_generator(self):
        return [self.d_layer_dict // (2**i) for i in range(self.cfg.n_nestings)]

    @cached_property
    def decoder(self):
        return MatryoshkaCLTDecoder(
            self.d_layer_data, self.d_layer_dict, self.nesting_boundaries, self.cfg
        )

    @cached_property
    def initializers(self):
        return self.init.split_initializer(self.cfg.n_sites)

    @cached_property
    def encoders(self):
        return [self.initializers[i].encoder.cuda() for i in range(self.cfg.n_sites)]

    def encode_up_to_layer(self, layer_idx: int):
        # Generates a model that breaks up an input of shape [batch_size, d_layer_data * (layer_idx + 1)],
        # and encodes it into [(layer_idx + 1), batch_size, d_layer_dict] shape.
        return ReuseForward(
            Seq(
                split=Lambda(lambda x: torch.chunk(x, layer_idx + 1, dim=-1)),
                encode=cl.Router(
                    *[self.encoders[i] for i in range(layer_idx + 1)]
                ).reduce(lambda *x: torch.stack(x, dim=0)),
            )
        )

    @cached_property
    def pre_encoders(self):
        # saeco multisite prepares the input catted as
        # [batch_size, d_layer_data * n_sites] so we chunk it into n_sites pieces

        # The decoder will recieve the latents still chunked as
        # [n_sites, batch_size, d_layer_dict]
        return self.encode_up_to_layer(self.cfg.n_sites - 1)

    def reconstruct_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cache = SAECache()

        encodings = self.encode_up_to_layer(layer_idx)(x, cache=cache)
        recons = self.decoder.decode_single_layer(encodings, layer_idx)

        return recons
    def reconstruct_up_to_layer(sae:Trainable, read_sites:list[torch.Tensor], layer_idx: int, num_layers: int) -> torch.Tensor:
        cache = SAECache()

        assert len(read_sites) == layer_idx

        tensors = read_sites + [torch.zeros_like(read_sites[0])] * (num_layers - len(read_sites))
        results= sae(tensors, cache=cache)
        result = results.chunk(num_layers, dim=-1)[layer_idx]
        return result
    @cached_property
    def nonlinearity(self):
        return nn.ReLU()

    @model_prop
    def model(self):
        return SAE(
            encoder_pre=self.pre_encoders,
            nonlinearity=self.nonlinearity,
            decoder=self.decoder,
        )

    @loss_prop
    def l2_loss(self):
        return MatryoshkaLoss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)

    def get_evaluation_functions(
        self,
    ) -> dict[str, ReconstructionEvaluatorFunctionProtocol]:
        return {"recons/": get_multisite_recons_loss}

from typing import overload
class CLT_SAE(SAE):
    @overload
    def __init__(
        self,
        *,
        encoder_pre: Literal[None] = None,
        nonlinearity: Literal[None] = None,
        encoder: nn.Module,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...

    @overload
    def __init__(
        self,
        *,
        encoder_pre: nn.Module,
        nonlinearity: nn.Module,
        encoder: Literal[None] = None,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...
    def __init__(
        self,
        *,
        encoder_pre: nn.Module | None = None,
        nonlinearity: nn.Module | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ):
