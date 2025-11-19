from functools import cached_property

import torch
import torch.nn as nn

import saeco.components.features.features as ft
import saeco.core as cl
from saeco.architecture import SAE, Architecture, loss_prop, model_prop
from saeco.components import Lambda, Loss, SparsityPenaltyLoss
from saeco.components.features.features_param import FeaturesParam
from saeco.components.ops import Indexer
from saeco.components.sae_cache import SAECache
from saeco.core import Seq
from saeco.core.reused_forward import ReuseForward
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig


class MatryoshkaCLTConfig(SweepableConfig):
    n_sites: int = 12
    n_nestings: int = 3


class MatryoshkaLoss(Loss):
    def loss(self, x, y, y_pred, cache: SAECache):
        return torch.mean((y.unsqueeze(0) - y_pred) ** 2)


def multilayer_decoder(weight: torch.Tensor):
    def decode(x: torch.Tensor):
        # have to do the .to(x.dtype) bc something where the autocast doesn't work
        # didn't look into it enough to deeply understand it
        # einops.einsum(x, weight.to(x.dtype),"l b d, l o d -> b (l o)", )
        return torch.einsum("lbd,lod->bo", x, weight.to(x.dtype))
        # return einops.rearrange(o, "b l o -> b (l o)")

        return torch.einsum("lbd,lod->blo", x, weight.to(x.dtype))

        return torch.bmm(x, weight.transpose(-2, -1).to(x.dtype))

    return decode


def slice_dim(low, high, dim):
    return (slice(None),) * dim + (slice(low, high),)


def split_tensor(x, bounds, dim):
    bounds = [0] + bounds + [x.shape[dim]]
    if dim == -1:
        dim = x.dim()
    return [x[slice_dim(bounds[i], bounds[i + 1], dim)] for i in range(len(bounds) - 1)]


class SplittableDecoder(
    cl.Module, ft.OrthogonalizeFeatureGradsMixin, ft.NormFeaturesMixin
):
    def __init__(self, d_dict, num_layers, d_data, num_nestings):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_layers * d_data, d_dict))
        self.d_dict = d_dict
        self.n_nestings = num_nestings
        self.n_layers = num_layers
        self.d_data = d_data

    def forward(self, acts, *, cache: SAECache):
        splits = self.boundary_generator()
        splits.sort()
        return torch.stack(
            [
                dec(x_i)
                for i, (dec, x_i) in enumerate(
                    zip(self.split_dec(splits), split_tensor(acts, splits, dim=2))
                )
            ],
            dim=0,
        ).cumsum(dim=0)

    def split_dec(self, bounds: list[int]):
        return [
            multilayer_decoder(w)
            for w in split_tensor(
                self.weight.view(self.n_layers, self.d_data, self.d_dict), bounds, dim=2
            )
        ]

    def boundary_generator(self):
        return [self.d_dict // (2**i) for i in range(1, self.n_nestings)]

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
    def __init__(self, num_layers, d_data):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(num_layers * d_data))

    def forward(self, x, *, cache: SAECache):
        return x + self.bias


class MatryoshkaCLTDecoder(cl.Module):
    def __init__(self, d_data: int, d_dict: int, cfg: MatryoshkaCLTConfig):
        super().__init__()

        self.decode = Seq(
            route_to_decoders=cl.Parallel(
                *[Indexer.L[: i + 1] for i in range(cfg.n_sites)]
            ).reduce(lambda *x: x),
            splittable_decoders=cl.Router(
                *[
                    SplittableDecoder(
                        d_dict=d_dict,
                        num_layers=n + 1,
                        d_data=d_data,
                        num_nestings=cfg.n_nestings,
                    )
                    for n in range(cfg.n_sites)
                ]
            ).reduce(lambda *x: torch.cat(x, dim=-1)),
            decoder_bias=AddSplitDecoderBias(num_layers=cfg.n_sites, d_data=d_data),
        )

    def forward(self, x, *, cache: SAECache):
        decoded_layers = cache(self).decode(x)
        return decoded_layers


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
                ).reduce(lambda *x: torch.stack(x, dim=0)),
            )
        )

    # @cached_property
    # def encoder(self):
    #     return ReuseForward(
    #         Seq(
    #             pre_encs=self.pre_encoders,
    #             nonlinearity=nn.ReLU(),
    #         )
    #     )

    # def decoded_layer(self, layer: int, nl: int, nu: int):
    #     layer_acts = Seq(
    #         enc=self.encoder,
    #         layer_inputs=Indexer.L[: layer + 1],
    #     )

    def setup(self):
        assert self.init.d_dict % self.cfg.n_sites == 0
        assert self.init.d_data % self.cfg.n_sites == 0
        self.d_layer_dict = self.init.d_dict // self.cfg.n_sites
        self.d_layer_data = self.init.d_data // self.cfg.n_sites

        assert (2 ** (self.cfg.n_nestings - 1)) <= self.d_layer_dict

        self.nesting_sizes = [
            self.d_layer_dict // (2**i) for i in range(self.cfg.n_nestings)
        ]

    def generate_cross_layer_decode(self, nesting_size):
        assert nesting_size <= self.d_layer_dict
        return cl.Parallel(
            *[
                Seq(
                    Lambda(
                        lambda x, idx=i: torch.cat(
                            [
                                x[
                                    :,
                                    self.d_layer_dict * j : self.d_layer_dict * j
                                    + nesting_size,
                                ]
                                for j in range(idx + 1)
                            ],
                            dim=1,
                        )
                    ),
                    nn.Linear(
                        in_features=nesting_size * (i + 1),
                        out_features=self.d_layer_data,
                    ),
                )
                for i in range(self.cfg.n_sites)
            ]
        ).reduce(lambda *x: torch.cat(x, dim=1))

    # @cached_property
    # def encode_each_layer(self):

    #     encode_module = cl.Parallel(
    #         *[
    #             Seq(
    #                 Lambda(
    #                     lambda x, idx=i: x[
    #                         :, self.d_layer_data * idx : self.d_layer_data * (idx + 1)
    #                     ]
    #                 ),
    #                 nn.Linear(
    #                     in_features=self.d_layer_data,
    #                     out_features=self.d_layer_dict,
    #                 ),
    #             )
    #             for i in range(self.cfg.n_sites)
    #         ]
    #     ).reduce(lambda *x: torch.cat(x, dim=1))

    #     return Seq(
    #         weight=encode_module,
    #         nonlinearity=nn.ReLU(),  # Whatever nonlinearity you want
    #     )

    @cached_property
    def stacked_model(self):
        return SAE(
            encoder_pre=self.pre_encoders,
            nonlinearity=nn.ReLU(),
            decoder=self.decoder,
        )

    @model_prop
    def model(self):
        return self.stacked_model

    @loss_prop
    def l2_loss(self):
        return MatryoshkaLoss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
