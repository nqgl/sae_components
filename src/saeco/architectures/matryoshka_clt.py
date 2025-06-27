import random
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

from saeco.components.jumprelu import JumpReLU
from saeco.components.ops import Indexer
from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward

from saeco.data.model_cfg import ActsDataConfig
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

    per_decoder_split: bool = True


def reconstruct_clt_layer(sae: Trainable, mlp_inputs: list[torch.Tensor], num_layers):
    cache = SAECache()
    tensors = mlp_inputs + [
        torch.zeros_like(mlp_inputs[0]) for _ in range(num_layers - len(mlp_inputs))
    ]

    catted = torch.cat(tensors, dim=-1).float()

    results = sae(catted[0], cache=cache).to(mlp_inputs[0].dtype)[-1, ...].unsqueeze(0)
    result = results.chunk(num_layers, dim=-1)[len(mlp_inputs) - 1]

    return result


def with_sae_runner(
    model: nnsight.LanguageModel, encoder: Trainable, cfg: ActsDataConfig
):

    def saerunner(tokens):
        mlp_inputs = []

        with model.trace(tokens):
            for i in range(12):
                mlp_input_site = f"transformer.h.{i}.mlp.input"
                mlp_output_site = f"transformer.h.{i}.mlp.output"

                mlp_input = getsite(model, mlp_input_site).save()

                mlp_inputs.append(mlp_input)

                acts_re = nnsight.apply(reconstruct_clt_layer, encoder, mlp_inputs, 12)

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


def slice_dim(low, high, dim):
    return (slice(None),) * dim + (slice(low, high),)


def split_tensor(x, bounds, dim):
    if dim == -1:
        dim = x.dim()

    if bounds[0] != 0:
        bounds = [0] + bounds
    if bounds[-1] != x.shape[dim]:
        bounds = bounds + [x.shape[dim]]

    return [x[slice_dim(bounds[i], bounds[i + 1], dim)] for i in range(len(bounds) - 1)]


def generate_random_boundary(n_slices, max_index):
    return sorted(
        [0]
        + [random.randint(1, max_index - 1) for _ in range(n_slices - 1)]
        + [max_index]
    )


class SplittableDecoder(
    cl.Module, ft.OrthogonalizeFeatureGradsMixin, ft.NormFeaturesMixin
):
    def __init__(
        self, d_dict, num_layers, d_data, num_nestings, per_decoder_split=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_layers * d_dict, d_data))

        self.d_dict = d_dict
        self.n_nestings = num_nestings
        self.n_layers = num_layers
        self.d_data = d_data
        self.per_decoder_split = per_decoder_split

    def forward(self, input, *, cache: SAECache):
        # If a boundary is shared by all decoders, then the input is a tuple of (acts, splits)
        if self.per_decoder_split:
            splits = generate_random_boundary(self.n_nestings, self.d_dict)
            acts = input
        else:
            acts, splits = input

        return torch.stack(
            [
                torch.einsum("lbd,ldo->bo", x_i, dec_i)
                for (dec_i, x_i) in zip(
                    self.split_dec(splits), split_tensor(acts, splits, dim=2)
                )
            ],
            dim=0,
        ).cumsum(dim=0)

    def split_dec(self, bounds: list[int]):
        return split_tensor(
            einops.rearrange(
                self.weight, "(l d_dict) d_data -> l d_dict d_data", l=self.n_layers
            ),
            bounds,
            dim=1,
        )

    @cached_property
    def features(self):
        return {
            "weight": FeaturesParam(
                self.weight,
                feature_index=0,
                feature_parameter_type=FeaturesParam.FPTYPES.dec,
                param_id=f"{self.n_layers - 1}",
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
                d_dict=d_layer_dict,
                num_layers=n + 1,
                d_data=d_layer_data,
                num_nestings=cfg.n_nestings,
                per_decoder_split=cfg.per_decoder_split,
            )
            for n in range(cfg.n_sites)
        ]

        self.weights = [dec.weight for dec in self.decoders]

        self.decode = Seq(
            split_into_layers=Lambda(
                lambda x: einops.rearrange(
                    x, "batch (layer d_dict) -> layer batch d_dict", d_dict=d_layer_dict
                )
            ),
            route_to_decoders=cl.Parallel(
                *[Indexer.L[: i + 1] for i in range(cfg.n_sites)]
            ).reduce(lambda *x: x),
            **useif(
                not cfg.per_decoder_split,
                generate_boundary=Seq(
                    generate_splits=Lambda(
                        lambda x: (
                            x,
                            generate_random_boundary(cfg.n_nestings, d_layer_dict),
                        )
                    ),
                    prepare_splits=Lambda(lambda x: [(x_i, x[1]) for x_i in x[0]]),
                ),
            ),
            splittable_decoders=cl.Router(
                *[
                    SplittableDecoder(
                        d_dict=d_layer_dict,
                        num_layers=n + 1,
                        d_data=d_layer_data,
                        num_nestings=cfg.n_nestings,
                        per_decoder_split=cfg.per_decoder_split,
                    )
                    for n in range(cfg.n_sites)
                ]
            ).reduce(lambda *x: torch.cat(x, dim=-1)),
            decoder_bias=AddSplitDecoderBias(
                num_layers=cfg.n_sites, d_layer_data=d_layer_data
            ),
        )

    def forward(self, x, *, cache: SAECache):
        return cache(self).decode(x)


class MatryoshkaCLT(Architecture[MatryoshkaCLTConfig]):
    def boundary_generator(self):
        return [self.d_layer_dict // (2**i) for i in range(self.cfg.n_nestings)]

    @cached_property
    def decoder(self):
        return MatryoshkaCLTDecoder(self.d_layer_data, self.d_layer_dict, self.cfg)

    @cached_property
    def initializers(self):
        return self.init.split_initializer(self.cfg.n_sites)

    @cached_property
    def pre_encoders(self):
        return ReuseForward(
            Seq(
                split=Lambda(lambda x: torch.chunk(x, self.cfg.n_sites, dim=-1)),
                encode=cl.Router(
                    *[self.initializers[i].encoder for i in range(self.cfg.n_sites)]
                ).reduce(lambda *x: torch.cat(x, dim=-1)),
            )
        )

    def setup(self):
        assert self.init.d_dict % self.cfg.n_sites == 0
        assert self.init.d_data % self.cfg.n_sites == 0
        self.d_layer_dict = self.init.d_dict // self.cfg.n_sites
        self.d_layer_data = self.init.d_data // self.cfg.n_sites

        assert (2 ** (self.cfg.n_nestings - 1)) <= self.d_layer_dict

        self.nesting_boundaries = self.boundary_generator()

    @model_prop
    def model(self):
        return SAE(
            encoder_pre=self.pre_encoders,
            #            nonlinearity=JumpReLU(0.03, 1),
            nonlinearity=nn.ReLU(),
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
