import torch
import torch.nn as nn

import saeco.components as co
from saeco.components.features.linear_type import LinDecoder, LinEncoder
from saeco.core.basic_ops import Add
from saeco.initializer.linear_factory import LinearFactory


class Initializer:
    """Factory for an architecture's parameters, sized from ``InitConfig``.

    Exposed as ``self.init`` inside an ``Architecture``. Provides the
    encoder/decoder linears (``self.init.encoder`` / ``self.init.decoder``)
    and bias helpers so a model can be assembled without hand-wiring shapes.
    The per-side builders are reachable as ``self.init.enc_builder`` /
    ``self.init.dec_builder`` for configuration (bias presence, shape, weight
    tying) before the linears are materialized.

    By default the decoder is initialized as the transpose of the encoder
    (``tied_init``); the weights are independent thereafter.
    """

    def __init__(
        self,
        d_data: int,
        d_dict: int | None = None,
        dict_mult: int | float | None = None,
        l0_target: int | float | None = None,
    ):
        self.d_data = d_data
        d_dict = d_dict or int(d_data * dict_mult)
        self.d_dict = d_dict
        self.l0_target = int(l0_target) if l0_target is not None else None
        self.tied_init = True
        self.tied_weights = False
        self._decoder = LinearFactory(d_dict, d_data, cls=LinDecoder)
        self._encoder = LinearFactory(d_data, d_dict, cls=LinEncoder)

        if self.tied_init:
            self._decoder.tied_weights_init(self._encoder)
        if self.tied_weights:
            self._decoder.tie_weights(self._encoder)

    def get_hl_initializer(self, bf, l0_target=None) -> "Initializer":
        assert self.d_dict % bf == 0
        new = Initializer(
            d_data=self.d_data,
            d_dict=self.d_dict // bf,
            l0_target=l0_target or self.l0_target,
        )
        new.tied_init = self.tied_init
        new.tied_weights = self.tied_weights
        return new

    def split_initializer(
        self, bf, split_d_data=True, l0_target=None
    ) -> list["Initializer"]:
        assert self.d_dict % bf == 0
        if split_d_data:
            assert self.d_data % bf == 0
        out = []
        for _i in range(bf):
            new = Initializer(
                d_data=self.d_data // bf if split_d_data else self.d_data,
                d_dict=self.d_dict // bf,
                l0_target=l0_target or self.l0_target,
            )
            new.tied_init = self.tied_init
            new.tied_weights = self.tied_weights
            out.append(new)
        return out

    @property
    def enc_builder(self) -> LinearFactory:
        """The encoder *builder*: configure bias/shape/tie here, *before* the
        encoder linear is materialized via :attr:`encoder`.

        Accessing it once the encoder has been built raises — configuration at
        that point would be silently ignored, so read :attr:`encoder` instead.
        """
        if not self._encoder.unset:
            raise RuntimeError(
                "init.enc_builder is configure-before-build only: the encoder "
                "linear has already been built. Read init.encoder instead."
            )
        return self._encoder

    @property
    def dec_builder(self) -> LinearFactory:
        """The decoder *builder* — see :attr:`enc_builder`."""
        if not self._decoder.unset:
            raise RuntimeError(
                "init.dec_builder is configure-before-build only: the decoder "
                "linear has already been built. Read init.decoder instead."
            )
        return self._decoder

    @property
    def encoder(self) -> LinEncoder:
        enc = self._encoder.get()
        assert isinstance(enc, LinEncoder)
        return enc

    @property
    def decoder(self) -> LinDecoder:
        dec = self._decoder.get()
        assert isinstance(dec, LinDecoder)
        return dec

    def bias(self, d: int, name: str | None = None) -> nn.Parameter:
        return nn.Parameter(torch.zeros(d))

    def dict_bias(self) -> nn.Parameter:
        return self.bias(self.d_dict)

    def new_encoder_bias(self) -> co.EncoderBias:
        return co.EncoderBias(Add(self.bias(self.d_dict)))

    def new_decoder_bias(self) -> nn.Parameter:
        return self.bias(self.d_data)
