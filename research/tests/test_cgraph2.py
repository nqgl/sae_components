"""Tests for the cgraph2 graph-composition library + the gated_graph arch.

The demos under `saeco_research.cgraph2` assert their own invariants in `main()`;
we run them here so they're part of the CI suite, plus a few focused unit checks
and a small CPU integration test of the whole graph-native trainable.
"""

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    "module_name",
    [
        "saeco_research.cgraph2.demo",
        "saeco_research.cgraph2.gated",
        "saeco_research.cgraph2.funcstd_demo",
        "saeco_research.cgraph2.inherit_subslot_demo",
        "saeco_research.cgraph2.bubble_demo",
    ],
)
def test_cgraph2_demo_runs(module_name: str) -> None:
    importlib.import_module(module_name).main()


def test_interning_is_structural() -> None:
    import torch.nn as nn

    from saeco_research.cgraph2 import Input, op

    enc = nn.Linear(4, 4)
    # same producer + inputs -> the same interned node
    assert op(enc)(Input) is op(enc)(Input)
    # function structure, not identity: equivalent lambdas fuse
    assert op(lambda a: a + 1)(Input) is op(lambda b: b + 1)(Input)
    assert op(lambda a: a + 1)(Input) is not op(lambda a: a + 2)(Input)


def test_forward_values_runs_each_node_once() -> None:
    import torch.nn as nn

    import saeco.core as cl
    from saeco_research.cgraph2 import GraphSpec, Input, Op, op, op_slot

    class Counter(cl.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.n = 0

        def forward(self, x, *, cache, **kw):
            self.n += 1
            return self.lin(x)

    enc = Counter()

    class Spec(GraphSpec):
        encoder: Op = op_slot()
        h = encoder(Input)
        a = op(torch.relu)(h)
        b = op(torch.sigmoid)(h)
        output = a + b

    g = Spec(encoder=enc)
    with g.forward_values(torch.randn(2, 4), cache=cl.Cache()) as save:
        save(g.a)
        save(g.b)
        save(g.output)
    assert enc.n == 1  # shared encoder node ran once across all three saves


def test_arg_slots_are_independent() -> None:
    # regression: arg(default=Input) must be a *unique* slot, not the global Input
    # itself -- binding one slot must not rewrite the other or Input.
    import torch.nn as nn

    import saeco.core as cl
    from saeco_research.cgraph2 import Arg, GraphSpec, Input, arg, op

    class Two(GraphSpec):
        a: Arg = arg(default=Input)
        b: Arg = arg(default=Input)
        output = op(lambda x, y: x - y)(a, b)

    p = nn.Parameter(torch.ones(4))
    g = Two(a=p)  # a -> param; b stays Input
    out = g.out.build()(torch.full((4,), 3.0), cache=cl.Cache())
    # with the old bug both a and b (== Input) would bind to the param -> 0;
    # correct: a=param(1) - b=Input(3) == -2
    assert torch.allclose(out, torch.full((4,), -2.0))


def test_affine_normalizer_matches_generalized() -> None:
    # AffineNormalizer must reproduce StaticInvertibleGeneralizedNormalizer (the
    # stock static normalizer) for the static {DONTUSE, PRIMED} configs -- on the
    # default GNConfig *and* with a per-feature mean (mu_e=PRIMED).
    import saeco.core as cl
    from saeco.trainer.normalizers.normalizer import (
        Aggregation,
        GNConfig,
        StaticInvertibleGeneralizedNormalizer,
    )
    from saeco_research.cgraph2.sae import AffineNormalizer

    class _Init:
        d_data = 32

    class _B:
        def __init__(self, t):
            self.input = t

    torch.manual_seed(0)
    d = 32
    samples = [torch.randn(64, d) * 4 + 1 for _ in range(20)]

    for cfg in (GNConfig(), GNConfig(mu_e=Aggregation.PRIMED)):
        ref = StaticInvertibleGeneralizedNormalizer(_Init, cfg, eps=1e-7)
        ref.prime_normalizer(iter([_B(s) for s in samples]))
        aff = AffineNormalizer(d, cfg=cfg, eps=1e-7)
        aff.prime_normalizer(iter([_B(s) for s in samples]))

        x = torch.randn(10, d) * 4 + 1
        xn_ref = ref(x, cache=cl.Cache())
        xn_aff = aff(x, cache=cl.Cache())
        assert torch.allclose(xn_ref, xn_aff, atol=1e-5, rtol=1e-4), (
            float((xn_ref - xn_aff).abs().max())
        )
        # invert is a true round-trip and matches the stock denormalization
        xdn_aff = aff.invert(xn_aff)
        assert torch.allclose(xdn_aff, x, atol=1e-4)
        assert torch.allclose(ref.invert(xn_ref, cache=cl.Cache()), xdn_aff, atol=1e-5)


def test_affine_normalizer_rejects_dynamic_aggregations() -> None:
    from saeco.trainer.normalizers.normalizer import Aggregation, GNConfig
    from saeco_research.cgraph2.sae import AffineNormalizer

    with pytest.raises(NotImplementedError):
        AffineNormalizer(8, cfg=GNConfig(std_e=Aggregation.BATCH_AVG))


def test_gated_graph_trains_one_step_cpu() -> None:
    import saeco.components.features as ft
    from saeco.components.resampling.anthropic_resampling import (
        AnthResampler,
        AnthResamplerConfig,
    )
    from saeco.initializer.initializer import Initializer
    from saeco.trainer.train_cache import TrainCache
    from saeco_research.architectures.gated_graph.model import (
        GatedAuxSpec,
        GatedModelSpec,
    )
    from saeco_research.cgraph2.sae import (
        AffineNormalizer,
        FreqTracker,
        GraphL2Loss,
        GraphNormalizer,
        GraphSparsityLoss,
        GraphTrainable,
    )

    torch.manual_seed(0)
    d = 16
    init = Initializer(d, dict_mult=4, l0_target=8)
    init._encoder.bias = False
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    class _B:
        def __init__(self, t):
            self.input = t

    def buf():
        while True:
            yield _B(torch.randn(32, d) * 4 + 1)

    norm = AffineNormalizer(d)
    norm.prime_normalizer(buf())
    gn = GraphNormalizer(norm)

    enc = init.encoder
    gate_b, mag_b, log_r = (
        init.new_encoder_bias(),
        init.new_encoder_bias(),
        init.dict_bias(),
    )
    main = GatedModelSpec(
        encoder=enc, gate_bias=gate_b, mag_bias=mag_b, decoder=init.decoder, log_r=log_r
    )
    ftrack = FreqTracker(init.d_dict)
    aux = GatedAuxSpec(
        encoder=enc,
        gate_bias=gate_b,
        decoder=init._decoder.detached,
        freqs=ftrack,
    ).set_to_aux_model("model_aux")
    losses = {
        "L2_loss": GraphL2Loss.on(main),
        "L2_aux_loss": GraphL2Loss.on(aux),
        "sparsity_loss": GraphSparsityLoss.on(aux),
    }
    tr = GraphTrainable(
        [main, aux], AnthResampler(AnthResamplerConfig(expected_biases=2)), losses, gn
    )

    names = tr.get_losses_and_metrics_names()
    x = torch.randn(32, d) * 4 + 1
    coeffs = {"sparsity_loss": 3e-5, "L2_loss": 1}

    # the shared encoder runs once across main + aux + every loss in one pass
    calls = [0]
    enc.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    cache = TrainCache()
    cache._watch(names)
    tr.loss(x, cache=cache, y=x, coeffs=coeffs)
    assert calls[0] == 1
    for key in ("L2_loss", "L2_aux_loss", "sparsity_loss", "L0", "cosim", "loss"):
        assert cache.get(key) is not None  # the trainer-facing cache is populated

    # the freq tracker bubbled up: its EMA advanced (deferred update) and the
    # below_* freq metrics got written
    assert float(ftrack.freqs.sum()) > 0.0
    assert cache.get("below_3e-5") is not None

    # and it learns
    opt = torch.optim.Adam(tr.parameters(), lr=1e-3)
    hist = []
    for _ in range(40):
        c = TrainCache()
        c._watch(names)
        loss = tr.loss(x, cache=c, y=x, coeffs=coeffs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    assert hist[-1] < hist[0]
