from saeco.architectures.initialization.linear_factory import LinearFactory, Tied
import saeco.components as co
from saeco.core.basic_ops import Add


import torch
import torch.nn as nn


class Initializer:
    def __init__(
        self, d_data, d_dict=None, dict_mult=None, l0_target=None, median=None
    ):
        self.d_data = d_data
        d_dict = d_dict or d_data * dict_mult
        self.d_dict = d_dict
        self.l0_target = l0_target
        # self.tied_bias = True
        self.tied_init = True
        self.tied_weights = False
        self.encoder_init_weights = None

        self._decoder: LinearFactory = LinearFactory(
            d_dict, d_data, wrappers=[co.LinDecoder]
        )
        self._encoder: LinearFactory = LinearFactory(
            d_data, d_dict, wrappers=[co.LinEncoder]
        )
        if self.tied_init:
            self._decoder.tied_weights_init(self._encoder)
        if self.tied_weights:
            self._decoder.tie_weights(self._encoder)
        self.geo_med = median
        if median is not None:
            self._decoder._bias_tie = Tied(median, Tied.TO_VALUE, "bias")
        self._b_dec = None

    @property
    def encoder(self):
        return self._encoder.get()

    @property
    def decoder(self):
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
