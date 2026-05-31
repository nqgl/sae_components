from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


def _orient(dst: Tensor, src: Tensor) -> Tensor:
    """Return ``src`` oriented to match ``dst``'s shape.

    Encoder and decoder weights are transposes of each other, so tying one to
    the other means copying/sharing the transpose. When the two shapes are
    ambiguously compatible (i.e. square), prefer the transpose — matching the
    historical default for tied initialization.
    """
    if dst.shape == src.shape and dst.shape != src.transpose(-2, -1).shape:
        return src
    assert dst.shape == src.transpose(-2, -1).shape, (
        f"cannot tie a weight of shape {tuple(dst.shape)} to one of "
        f"shape {tuple(src.shape)}"
    )
    return src.transpose(-2, -1)


class LinearFactory:
    """Deferred builder for one of an SAE's encoder/decoder linears.

    Configure shape (``d_in``/``d_out``), ``bias`` presence, and an optional
    weight/bias initialization policy (tie, copy-transpose, or set-to-value)
    *before* the linear is first materialized via :meth:`get`; afterwards the
    built linear is cached and further configuration is rejected. The produced
    module is the concrete ``cls`` (``LinEncoder`` / ``LinDecoder``), so it
    carries the feature-parameter metadata that resampling and the feature
    hooks rely on.
    """

    def __init__(self, d_in, d_out, cls: type[nn.Linear] = nn.Linear, bias=True):
        self.d_in = d_in
        self.d_out = d_out
        self.cls = cls
        self._bias = bias
        self._linear: nn.Linear | None = None
        self._weight_init: Callable[[nn.Linear], None] | None = None
        self._bias_init: Callable[[nn.Linear], None] | None = None

    @property
    def unset(self) -> bool:
        return self._linear is None

    def _assert_configurable(self) -> None:
        if not self.unset:
            raise ValueError("Cannot configure the linear after it has been built")

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._assert_configurable()
        self._bias = value

    def make_new(self) -> nn.Linear:
        lin = self.cls(self.d_in, self.d_out, bias=self.bias)
        if self._weight_init is not None:
            self._weight_init(lin)
        if self.bias and self._bias_init is not None:
            self._bias_init(lin)
        return lin

    def get(self) -> nn.Linear:
        if self._linear is None:
            self._linear = self.make_new()
        return self._linear

    @property
    def raw(self) -> nn.Linear:
        return self.get()

    # --- weight/bias initialization policies (configure before build) ---

    def tied_weights_init(self, other: "LinearFactory") -> None:
        """Initialize the weight as a (transposed) copy of ``other``'s."""
        self._assert_configurable()
        assert self._weight_init is None

        def init(lin):
            lin.weight.data[:] = _orient(lin.weight.data, other.raw.weight.data)

        self._weight_init = init

    def tie_weights(self, other: "LinearFactory") -> None:
        """Share weights with ``other`` (a transposed view — same storage)."""
        self._assert_configurable()
        assert self._weight_init is None

        def init(lin):
            lin.weight.data = _orient(lin.weight.data, other.raw.weight.data)

        self._weight_init = init

    def untie(self) -> None:
        """Clear any weight tie/init policy. Must precede the build."""
        self._assert_configurable()
        self._weight_init = None

    def init_weight_to(self, value) -> None:
        """Initialize the weight to a fixed tensor at build time.

        Covers the power-user pattern of building one side, then initializing
        the other from a tensor computed off it. Must precede the build.
        """
        self._assert_configurable()
        assert self._weight_init is None

        def init(lin):
            lin.weight.data[:] = value

        self._weight_init = init

    def const_init_bias(self, const=0) -> None:
        """Initialize the bias to a constant at build time."""
        self._assert_configurable()
        assert self._bias_init is None

        def init(lin):
            lin.bias.data[:] = const

        self._bias_init = init
