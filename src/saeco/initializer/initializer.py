import torch
import torch.nn as nn

import saeco.components as co
from saeco.components.features.linear_type import LinDecoderMixin, LinEncoderMixin
from saeco.core.basic_ops import Add
from saeco.initializer.linear_factory import LinearFactory, Tied
from .initializer_config import InitConfig


class Initializer:
    def __init__(
        self,
        d_data,
        d_dict=None,
        dict_mult=None,
        l0_target=None,
        median=None,
        weight_scale=None,
    ):
        self.d_data = d_data
        d_dict = d_dict or int(d_data * dict_mult)
        self.d_dict = d_dict
        self.l0_target = int(l0_target) if l0_target is not None else None
        # self.tied_bias = True
        self.tied_init = True
        self.tied_weights = False
        self.encoder_init_weights = None
        self._decoder: LinearFactory = LinearFactory(
            d_dict, d_data, wrappers=[], mixins=[LinDecoderMixin]
        )
        self._encoder: LinearFactory = LinearFactory(
            d_data, d_dict, wrappers=[], mixins=[LinEncoderMixin]
        )

        if self.tied_init:
            self._decoder.tied_weights_init(self._encoder)
        if self.tied_weights:
            self._decoder.tie_weights(self._encoder)
        self.geo_med = median
        if median is not None:
            self._decoder._bias_tie = Tied(median, Tied.TO_VALUE, "bias")
        self._b_dec = None
        if weight_scale is not None:
            assert self._encoder._weight_tie is None
            self._encoder._weight_tie = Tied(
                lambda x: x * weight_scale, Tied.INIT_FN, "weight"
            )

    def get_hl_initializer(self, bf, l0_target=None) -> "Initializer":
        assert self.d_dict % bf == 0
        new = Initializer(
            d_data=self.d_data,
            d_dict=self.d_dict // bf,
            l0_target=l0_target or self.l0_target,
        )
        # self.tied_bias = True
        new.tied_init = self.tied_init
        new.tied_weights = self.tied_weights
        new.encoder_init_weights = self.encoder_init_weights
        return new

    def split_initializer(
        self, bf, split_d_data=True, l0_target=None
    ) -> list["Initializer"]:
        assert self.d_dict % bf == 0
        if split_d_data:
            assert self.d_data % bf == 0
        l = []
        for i in range(bf):
            new = Initializer(
                d_data=self.d_data // bf if split_d_data else self.d_data,
                d_dict=self.d_dict // bf,
                l0_target=l0_target or self.l0_target,
            )
            # self.tied_bias = True
            new.tied_init = self.tied_init
            new.tied_weights = self.tied_weights
            new.encoder_init_weights = self.encoder_init_weights
            l.append(new)
        return l

    def split_clt_initializer(
        self, bf, split_d_data=True, l0_target=None
    ) -> list["Initializer"]:
        assert self.d_dict % bf == 0
        l = []
        for i in range(bf):
            new = Initializer(
                d_data=self.d_data // bf if split_d_data else self.d_data,
                d_dict=self.d_dict // bf,
                l0_target=l0_target or self.l0_target,
            )
            # self.tied_bias = True
            new.tied_init = self.tied_init
            new.tied_weights = self.tied_weights
            new.encoder_init_weights = self.encoder_init_weights
            l.append(new)
        return l

    @property
    def encoder(self) -> co.LinEncoder:
        return self._encoder.get()

    @property
    def decoder(self) -> co.LinDecoder:
        return self._decoder.get()

    @property
    def b_dec(self):
        if self._decoder.bias:
            return self._decoder.get().bias
        if self._b_dec is None:
            self._b_dec = self.data_bias()
            if self.geo_med is not None:
                self._b_dec.data[:] = self.geo_med
        return self._b_dec

    def bias(self, d, name=None):
        return nn.Parameter(torch.zeros(d))

    def dict_bias(self):
        return self.bias(self.d_dict)

    def new_encoder_bias(self):
        return co.EncoderBias(Add(self.bias(self.d_dict)))

    def data_bias(self):
        return self.bias(self.d_data)
