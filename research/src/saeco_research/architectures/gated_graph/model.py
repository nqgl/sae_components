"""The Gated SAE on cgraph v2, assembled exactly like the stock gated arch.

This mirrors ``src/saeco/architectures/gated/arch.py`` part-for-part: ``setup``
attaches the decoder wrappers, ``@model_prop`` / ``@aux_model_prop`` build the
main and aux autoencoders (as ``SAEGraphSpec`` model graphs), and ``@loss_prop``s
declare the losses. The model is authored as a ``GraphSpec`` class-block (the v2
authoring surface), not a hand-wired ``op(...)`` chain.

The shared encoder is the whole point: ``EncBlock``'s ``encoder(x)`` is one
interned node, and because the main and aux specs bind it (and the gate bias) to
the *same* modules, the gate path is literally the same node across the main
model, the aux model, and the sparsity loss -- it runs once, with no
``ReuseForward``.
"""

from functools import cached_property

import torch

import saeco.components.features as ft
from saeco.architecture import aux_model_prop, loss_prop, model_prop
from saeco.architecture.sae_architecture import Architecture
from saeco.components.resampling.anthropic_resampling import AnthResampler
from saeco_research.cgraph2 import Arg, Input, Op, arg, op, op_slot
from saeco_research.cgraph2.sae import (
    AffineNormalizer,
    FreqTracker,
    GraphL2Loss,
    GraphNormalizer,
    GraphSparsityLoss,
    GraphTrainable,
    SAEGraphSpec,
    metered,
)
from sweepable import SweepableConfig


class GatedGraphConfig(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True


def _thresh(t):
    return (t > 0).float()


# --- the model, as GraphSpec class-blocks (SAEGraphSpec = a model graph) -------
class GatedModelSpec(SAEGraphSpec):
    """main model: acts = relu(W x * exp(r) + b_mag) * 1[relu(W x + b_gate) > 0]."""

    encoder: Op = op_slot()
    gate_bias: Op = op_slot()
    mag_bias: Op = op_slot()
    decoder: Op = op_slot()
    log_r: Arg = arg()
    x: Arg = arg(default=Input)

    h = encoder(x)
    gate_acts = op(torch.relu)(gate_bias(h))
    mag = op(torch.relu)(mag_bias(h * op(torch.exp)(log_r)))
    acts = metered(mag * op(_thresh)(gate_acts))  # metrics site bubbles up L0/L1
    recon = decoder(acts)
    output = recon


class GatedAuxSpec(SAEGraphSpec):
    """aux model: acts = relu(W x + b_gate); recon = decoder(acts)."""

    encoder: Op = op_slot()
    gate_bias: Op = op_slot()
    decoder: Op = op_slot()
    freqs: Op = op_slot()  # a FreqTracker passthrough -- bubbles up the EMA freqs
    x: Arg = arg(default=Input)

    acts = freqs(op(torch.relu)(gate_bias(encoder(x))))
    recon = decoder(acts)
    output = recon


class GatedGraphSAE(Architecture[GatedGraphConfig]):
    def setup(self):
        # encoder shared across mag/gate/aux by interning, not ReuseForward.
        self.init._encoder.bias = False
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    # biases: gate_bias is shared between the main and aux models (so the gate
    # path is one node across both); mag_bias / log_r are main-only.
    @cached_property
    def _gate_bias(self):
        return self.init.new_encoder_bias()

    @cached_property
    def _mag_bias(self):
        return self.init.new_encoder_bias()

    @cached_property
    def _log_r(self):
        return self.init.dict_bias()

    @cached_property
    def _freqs(self):  # EMA firing-frequency tracker on the gate/aux acts
        return FreqTracker(self.init.d_dict)

    @model_prop
    def gated_model(self) -> GatedModelSpec:
        if self.cfg.pre_bias:
            raise NotImplementedError("pre_bias variant not yet ported to cgraph")
        return GatedModelSpec(
            encoder=self.init.encoder,
            gate_bias=self._gate_bias,
            mag_bias=self._mag_bias,
            decoder=self.init.decoder,
            log_r=self._log_r,
        )

    @aux_model_prop
    def model_aux(self) -> GatedAuxSpec:
        decoder = self.init._decoder.detached if self.cfg.detach else self.init.decoder
        return GatedAuxSpec(
            encoder=self.init.encoder,
            gate_bias=self._gate_bias,
            decoder=decoder,
            freqs=self._freqs,
        )

    @loss_prop
    def L2_loss(self):  # noqa: N802  # name is the train_cfg.coeffs key
        return GraphL2Loss.on(self.gated_model)

    @loss_prop
    def L2_aux_loss(self):  # noqa: N802
        return GraphL2Loss.on(self.model_aux)

    @loss_prop
    def sparsity_loss(self):
        return GraphSparsityLoss.on(self.model_aux)

    # -- trainable / normalizer assembly, mirroring Architecture.make_trainable -
    def make_resampler(self) -> AnthResampler:
        return AnthResampler(self.run_cfg.resampler_config)

    @cached_property
    def normalizer(self) -> GraphNormalizer:  # type: ignore[override]  # graph-native
        norm = AffineNormalizer(
            self.init.d_data, cfg=self.run_cfg.normalizer_cfg
        ).to(device=self.device)
        if self.state_dict is None:
            norm.prime_normalizer(self.data)
        return GraphNormalizer.of(norm)

    def make_trainable(self) -> GraphTrainable:
        if not self._instantiated:
            self.instantiate()
        trainable = GraphTrainable(
            [self._core_model, *self.aux_models],
            resampler=self.make_resampler(),
            losses=self._losses,
            normalizer=self.normalizer,
        )
        if self.state_dict is not None:
            print("loaded:", trainable.load_state_dict(self.state_dict))
        return trainable
