"""Tests for the cgraph2 graph-composition library + the gated_graph arch.

The demos under `saeco_research.cgraph2` assert their own invariants in `main()`;
we run them here so they're part of the CI suite, plus a few focused unit checks
and a small CPU integration test of the whole graph-native trainable.
"""

import importlib

import pytest
import torch


def _cgraph2_double(x):
    return x * 2


def _cgraph2_mul(a, b):
    return a * b


@pytest.mark.parametrize(
    "module_name",
    [
        "saeco_research.cgraph2.demo",
        "saeco_research.cgraph2.gated",
        "saeco_research.cgraph2.funcstd_demo",
        "saeco_research.cgraph2.inherit_subslot_demo",
        "saeco_research.cgraph2.bubble_demo",
        "saeco_research.cgraph2.multi_input_demo",
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
    from saeco_research.cgraph2 import GraphSpec, Input, Slot, op, op_slot

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
        encoder: Slot = op_slot()
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


def test_typed_dictbatch_fields_and_overrides() -> None:
    # DictBatch: class-level field is a FieldRef (spec-building handle); an
    # instance exposes the value; a missing declared field fails loud; and a
    # field can be an overridable slot (arg(default=field(...))).
    from saeco_research.cgraph2 import DictBatch, FieldRef, GraphSpec, arg, field, op
    from saeco_research.cgraph2.spec import Arg

    class Batch(DictBatch):
        input: torch.Tensor
        target: torch.Tensor

    # class access -> FieldRef carrying the name; field() reads it (== field("name"))
    assert isinstance(Batch.input, FieldRef)
    assert Batch.input.name == "input"
    assert field(Batch.input) is field("input")

    b = Batch(input=torch.ones(2, 3), target=torch.zeros(2, 3))
    assert torch.equal(b.input, torch.ones(2, 3))  # instance access -> value

    incomplete = Batch(input=torch.ones(2, 3))  # type: ignore[call-arg]
    assert incomplete.missing_fields() == ["target"]
    with pytest.raises(AttributeError):  # accessing an unprovided field is loud
        _ = incomplete.target

    class Sub(GraphSpec):
        x: Arg = arg(default=field(Batch.input))  # overridable; defaults to batch
        output = op(torch.relu)(x)

    import saeco.core as cl

    # default path: reads the batch's input field
    assert torch.equal(
        Sub().forward(b, cache=cl.Cache()), torch.ones(2, 3)
    )
    # override path: bind x to a fixed value, bypassing the batch
    fixed = torch.full((2, 3), -5.0)
    assert torch.equal(
        Sub(x=fixed).forward(b, cache=cl.Cache()), torch.zeros(2, 3)
    )


def test_field_slot_override_ignores_batch_field() -> None:
    # Overriding a field-slot must win over the batch: the field is read from the
    # binding, not the batch -- whether or not the batch carries that field.
    import saeco.core as cl
    from saeco_research.cgraph2 import DictBatch, GraphSpec, arg, field
    from saeco_research.cgraph2.core import _Field, _toposort
    from saeco_research.cgraph2.spec import Arg

    class B(DictBatch):
        a: torch.Tensor
        b: torch.Tensor

    class G(GraphSpec):
        a = field(B.a)
        b: Arg = arg(default=field(B.b))  # overridable; defaults to batch.b
        output = a + b

    g = G(b=torch.full((3,), 100.0))  # override b
    # override wins even when the batch lacks b...
    out_missing = g.forward(B(a=torch.ones(3)), cache=cl.Cache())  # type: ignore[call-arg]
    assert torch.equal(out_missing, torch.full((3,), 101.0))
    # ...and even when the batch *has* b (it must be ignored)
    full_batch = B(a=torch.ones(3), b=torch.full((3,), -999.0))
    out_present = g.forward(full_batch, cache=cl.Cache())
    assert torch.equal(out_present, torch.full((3,), 101.0))
    # structurally: field(B.b) is gone from the bound cone
    nodes = _toposort(*g._targets())
    assert not any(
        isinstance(n.producer, _Field) and n.producer.name == "b" for n in nodes
    )
    # default (no override) reads the batch
    assert torch.equal(
        G().forward(B(a=torch.ones(3), b=torch.full((3,), 5.0)), cache=cl.Cache()),
        torch.full((3,), 6.0),
    )


def test_partial_application() -> None:
    import saeco.core as cl
    from saeco_research.cgraph2 import GraphSpec, Input, arg, op
    from saeco_research.cgraph2.spec import Arg

    def _mul(a, b):
        return a * b

    def _add(a, b):
        return a + b

    class Denorm(GraphSpec):
        value: Arg = arg()
        scale: Arg = arg()
        shift: Arg = arg()
        output = op(_add)(op(_mul)(value, scale), shift)

    p = Denorm.partial(
        scale=op(lambda x: x * 0 + 2.0)(Input),
        shift=op(lambda x: x * 0 + 1.0)(Input),
    )
    p2 = p.rebind({Input: op(lambda b: b["input"])(Input)})
    done = p2(op(lambda b: b["recon"])(Input))

    out = done.forward(
        {"input": torch.tensor([9.0]), "recon": torch.tensor([5.0])},
        cache=cl.Cache(),
    )
    assert torch.allclose(out, torch.tensor([11.0]))

    out2 = done.forward(
        {"input": torch.tensor([100.0]), "recon": torch.tensor([5.0])},
        cache=cl.Cache(),
    )
    assert torch.allclose(out2, torch.tensor([11.0]))


def test_partial_completed_exposes_output_and_rebinds_defaults() -> None:
    # regression for two partial-application bugs:
    #  (1) a completed partial must expose `.output` (not just `.out`).
    #  (2) rebind() must rebase a still-open slot's ValueRef default, so a default
    #      like arg(default=Input) follows the same rebase as the rest.
    import saeco.core as cl
    from saeco_research.cgraph2 import GraphSpec, Input, arg, field, op
    from saeco_research.cgraph2.spec import Arg

    def _add(a, b):
        return a + b

    class G(GraphSpec):
        a: Arg = arg(default=Input)  # open; defaults to the (rebindable) Input
        b: Arg = arg()
        output = op(_add)(a, b)

    # (1) completed partial exposes `.output` (both slots bound to constants)
    done = G.partial(b=torch.tensor([4.0])).complete(a=torch.tensor([3.0]))
    assert done.output is not None
    out = done.forward(torch.zeros(1), cache=cl.Cache())  # a/b are constants
    assert torch.allclose(out, torch.tensor([7.0]))

    # (2) close `b` over a constant; `a` stays open with its Input default. Rebind
    # the Input that default rests on, then complete (a uses its *rebound* default).
    # A stale default would feed the whole batch dict into _add and blow up.
    p = G.partial(b=torch.tensor([10.0]))
    done2 = p.rebind({Input: field("x")}).complete()  # a resolves to field("x")
    out2 = done2.forward({"x": torch.tensor([3.0])}, cache=cl.Cache())
    assert torch.allclose(out2, torch.tensor([13.0]))  # a=x=3, b=10


def test_embedded_partial_subgraph() -> None:
    import saeco.core as cl
    from saeco_research.cgraph2 import GraphSpec, Input, arg, field, op, sub_slot
    from saeco_research.cgraph2.spec import Arg

    class Inner(GraphSpec):
        value: Arg = arg()
        scale: Arg = arg()
        output = op(_cgraph2_mul)(value, scale)

    class Outer(GraphSpec):
        s = op(_cgraph2_double)(Input)
        finish = Inner.partial(scale=s)
        output = Input

    class User(GraphSpec):
        outer: Outer = sub_slot()
        x_in = field("input")
        x_other = field("other")
        recon = field("recon")
        alt = field("alt")
        out = outer(x_in).finish(recon).output
        out_alt = outer(x_in).finish(alt).output
        out_other = outer(x_other).finish(recon).output
        output = out

    g = User(outer=Outer())
    assert g.out is not g.out_alt
    assert g.out is not g.out_other
    batch = {
        "input": torch.tensor([2.0, 3.0]),
        "other": torch.tensor([5.0, 7.0]),
        "recon": torch.tensor([11.0, 13.0]),
        "alt": torch.tensor([17.0, 19.0]),
    }

    with g.forward_values(batch, cache=cl.Cache()) as save:
        out = save(g.out)
        out_alt = save(g.out_alt)
        out_other = save(g.out_other)

    assert torch.allclose(out, batch["recon"] * 2 * batch["input"])
    assert torch.allclose(out_alt, batch["alt"] * 2 * batch["input"])
    assert torch.allclose(out_other, batch["recon"] * 2 * batch["other"])

    direct = g.outer(field("other")).finish(field("alt")).output
    direct_out = GraphSpec.of(output=direct).forward(batch, cache=cl.Cache())
    assert torch.allclose(direct_out, batch["alt"] * 2 * batch["other"])


def test_render_shows_deduped_shared_nodes() -> None:
    # render() is the flat execution view: a node shared by several handles
    # appears exactly once (the fusion made visible).
    import torch.nn as nn

    from saeco_research.cgraph2 import GraphSpec, Input, Slot, op, op_slot

    class Spec(GraphSpec):
        encoder: Slot = op_slot()
        h = encoder(Input)  # one node, read by both branches below
        a = op(torch.relu)(h)
        b = op(torch.sigmoid)(h)
        output = a + b

    r = Spec(encoder=nn.Linear(4, 4)).render()
    assert r.count("= Linear") == 1  # the shared encoder node is printed once
    for handle in ("h", "a", "b", "output"):
        assert handle in r  # handles are named/tagged in the view


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


def test_graph_normalizer_embeds_its_inverse() -> None:
    # D-12: GraphNormalizer embeds a Denormalizer partial closed over its primed
    # scale/shift; completing it (gn.denormalize_value(v)) must equal the wrapped
    # AffineNormalizer.invert, round-trip to identity, and stably cons (so forward
    # + inverse fuse) -- while a *different* normalizer's inverse shares no nodes.
    import saeco.core as cl
    from saeco_research.cgraph2.core import Input
    from saeco_research.cgraph2.sae import AffineNormalizer, GraphNormalizer

    class _B:
        def __init__(self, t):
            self.input = t

    def buf(d):
        while True:
            yield _B(torch.randn(32, d) * 3 + 2)

    torch.manual_seed(0)
    d = 8
    norm = AffineNormalizer(d)
    norm.prime_normalizer(buf(d))
    gn = GraphNormalizer(normalizer=norm)
    x = torch.randn(5, d) * 2 - 1

    # denormalize_value(Input) == AffineNormalizer.invert (the embedded inverse)
    got = gn.denormalize_value(Input).build()(x, cache=cl.Cache())
    assert torch.allclose(got, norm.invert(x), atol=1e-6)

    # round-trip: denormalize(normalize(x)) == x
    roundtrip = gn.denormalize_value(gn.normalized).build()(x, cache=cl.Cache())
    assert torch.allclose(roundtrip, x, atol=1e-5)

    # the scale/shift readers are stably consed (keyed by id(normalizer)), so the
    # completed inverse is one interned node -- two completions are the same object.
    assert gn.denormalize_value(Input) is gn.denormalize_value(Input)
    # a distinct normalizer's inverse is a distinct node (no cross-leak).
    norm2 = AffineNormalizer(d)
    norm2.prime_normalizer(buf(d))
    gn2 = GraphNormalizer(normalizer=norm2)
    assert gn2.denormalize_value(Input) is not gn.denormalize_value(Input)
    got2 = gn2.denormalize_value(Input).build()(x, cache=cl.Cache())
    assert torch.allclose(got2, norm2.invert(x), atol=1e-6)


def test_module_slot_forward_and_param_reads() -> None:
    # module_slot() plugs a leaf cl.Module in as a Slot descriptor -- m(x) is its
    # forward, m.get(lambda n: n.w) reads a buffer as a typed graph ref (consed by
    # module identity); on the instance m is the bound module. A typo'd attr fails
    # the type checker (statically) and loud at eval (lazily) -- never a wrong value.
    import saeco.core as cl
    from saeco_research.cgraph2.core import Input
    from saeco_research.cgraph2.spec import GraphSpec, Slot, module_slot

    class Scale(cl.Module):
        def __init__(self, w):
            super().__init__()
            self.register_buffer("w", torch.tensor(w))

        def forward(self, x, *, cache, **kw):
            return x * self.w

    class G(GraphSpec):
        m: Slot[Scale] = module_slot()
        out = m(Input)
        w = m.get(lambda n: n.w)
        output = out

    s = Scale([2.0, 3.0])
    g = G(m=s)
    x = torch.tensor([1.0, 1.0])
    assert g.out.build()(x, cache=cl.Cache()).tolist() == [2.0, 3.0]  # forward x*w
    assert g.w.build()(x, cache=cl.Cache()).tolist() == [2.0, 3.0]  # buffer read
    assert g.m is s  # the module is a navigable child
    assert G(m=s).w is g.w  # same module -> the reader node is consed
    assert G(m=Scale([2.0, 3.0])).w is not g.w  # distinct module -> distinct node

    # an attr the module doesn't have is type-illegal (pyright); at runtime the read
    # is lazy and fails loud at eval (never a silent wrong value).
    class Bad(GraphSpec):
        m: Slot[Scale] = module_slot()
        nope = m.get(lambda n: n.not_a_buffer)  # type: ignore[attr-defined]
        output = m(Input)

    with pytest.raises(AttributeError):
        Bad(m=s).nope.build()(x, cache=cl.Cache())


def test_affine_normalizer_rejects_dynamic_aggregations() -> None:
    from saeco.trainer.normalizers.normalizer import Aggregation, GNConfig
    from saeco_research.cgraph2.sae import AffineNormalizer

    with pytest.raises(NotImplementedError):
        AffineNormalizer(8, cfg=GNConfig(std_e=Aggregation.BATCH_AVG))


def _build_gated_trainable(d=16):
    import saeco.components.features as ft
    from saeco.components.resampling.anthropic_resampling import (
        AnthResampler,
        AnthResamplerConfig,
    )
    from saeco.initializer.initializer import Initializer
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
    enc = init.encoder
    gate_b, mag_b, log_r = (
        init.new_encoder_bias(),
        init.new_encoder_bias(),
        init.dict_bias(),
    )
    main = GatedModelSpec(
        encoder=enc, gate_bias=gate_b, mag_bias=mag_b, decoder=init.decoder, log_r=log_r
    )
    aux = GatedAuxSpec(
        encoder=enc,
        gate_bias=gate_b,
        decoder=init._decoder.detached,
        freqs=FreqTracker(init.d_dict),
    ).set_to_aux_model("model_aux")
    losses = {
        "L2_loss": GraphL2Loss.on(main),
        "L2_aux_loss": GraphL2Loss.on(aux),
        "sparsity_loss": GraphSparsityLoss.on(aux),
    }
    tr = GraphTrainable(
        [main, aux], AnthResampler(AnthResamplerConfig(expected_biases=2)),
        losses, GraphNormalizer(normalizer=norm),
    )
    return tr, d


def test_sub_slots_deferred_reduce() -> None:
    # sub_slots() + .reduce(per_item, combine): a class body folds a *dynamic* dict
    # of sub-graphs into one flat node, bound at construction.
    import saeco.core as cl
    from saeco_research.cgraph2 import GraphSpec, field, op, reduce_sum, sub_slots
    from saeco_research.cgraph2.core import _toposort

    class Loss(GraphSpec):
        x = field("v")
        out = op(lambda a: a.sum())(x)
        output = out

    class Total(GraphSpec):
        losses: dict = sub_slots()
        total = losses.reduce(lambda k, loss: field(f"w_{k}") * loss.out, reduce_sum)
        output = total

    g = Total(losses={"a": Loss(), "b": Loss()})
    batch = {"v": torch.tensor([1.0, 2.0, 3.0]), "w_a": 2.0, "w_b": 3.0}
    out = g.forward(batch, cache=cl.Cache())
    assert abs(float(out) - 30.0) < 1e-5  # (2 + 3) * sum([1,2,3])
    assert sorted(g.losses) == ["a", "b"]  # the bound dict is a navigable child
    # the fold is ONE flat N-ary node, not a pairwise tree
    nodes = _toposort(g.total)
    n_reduce = sum(1 for n in nodes if getattr(n.producer, "label", "") == "_stack_sum")
    assert n_reduce == 1


def test_total_loss_handle_weights_and_transcoder() -> None:
    from saeco.trainer.train_cache import TrainCache

    torch.manual_seed(0)
    tr, d = _build_gated_trainable()
    names = tr.get_losses_and_metrics_names()
    coeffs = {"sparsity_loss": 3e-5, "L2_loss": 1.0, "L2_aux_loss": 1.0}
    x = torch.randn(32, d) * 4 + 1

    # total_loss (a graph handle, weights = fields) == sum of weighted terms
    cache = TrainCache()
    cache._watch(names)
    total = tr.loss(x, cache=cache, y=x, coeffs=coeffs)
    expected = sum(coeffs[k] * float(cache.get(k)) for k in tr.graph.loss_terms)
    assert abs(float(total) - expected) < 1e-3

    # transcoder: target != input flows through (different recon target -> diff L2)
    c_ae, c_tc = TrainCache(), TrainCache()
    c_ae._watch(names)
    c_tc._watch(names)
    tr.loss(x, cache=c_ae, y=x, coeffs=coeffs)  # autoencoder: target == input
    tr.loss(x, cache=c_tc, y=torch.randn(32, d), coeffs=coeffs)  # target != input
    assert abs(float(c_ae.get("L2_loss")) - float(c_tc.get("L2_loss"))) > 1e-4


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
    gn = GraphNormalizer(normalizer=norm)

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


def test_deferred_execution_training_gate() -> None:
    # A @deferred_execution method runs at block exit on a *training* pass (skipped
    # on eval), and immediately when there's no active forward_values session.
    import saeco.core as cl
    from saeco_research.cgraph2.core import Input, deferred_execution, op
    from saeco_research.cgraph2.spec import GraphSpec

    class Counter:
        def __init__(self) -> None:
            self.n = 0

        @deferred_execution
        def bump(self, by) -> None:
            self.n += by

    class G(GraphSpec):
        output = op(lambda x: x)(Input)

    g = G()
    c = Counter()
    with g.forward_values(torch.tensor(1.0), cache=cl.Cache(), training=True) as save:
        save(g.output)
        c.bump(5)
        assert c.n == 0  # deferred -- not run yet, inside the block
    assert c.n == 5  # ran at block exit (training pass)

    with g.forward_values(torch.tensor(1.0), cache=cl.Cache(), training=False) as save:
        save(g.output)
        c.bump(5)
    assert c.n == 5  # skipped on an eval pass

    c.bump(3)  # no active session -> runs immediately
    assert c.n == 8


def test_freq_tracker_eval_pass_does_not_advance_ema() -> None:
    # B: the EMA writeback is a training callback, so an eval-mode forward_values
    # pass must not advance freqs; a training pass does.
    import saeco.core as cl
    from saeco_research.cgraph2.core import Input, op
    from saeco_research.cgraph2.sae import FreqTracker
    from saeco_research.cgraph2.spec import GraphSpec, Slot, op_slot

    ft = FreqTracker(8)

    class G(GraphSpec):
        enc: Slot = op_slot()
        acts = op(ft)(op(torch.relu)(enc(Input)))
        output = acts

    g = G(enc=torch.nn.Linear(8, 8))
    site = g.collect(FreqTracker)[0]
    x = torch.randn(16, 8)

    with g.forward_values(x, cache=cl.Cache(), training=False) as save:
        save(g.acts)
        ft.update(ft.updated(site.inputs[0]))
    assert float(ft.freqs.sum()) == 0.0  # eval: EMA did not advance

    with g.forward_values(x, cache=cl.Cache(), training=True) as save:
        save(g.acts)
        ft.update(ft.updated(site.inputs[0]))
    assert float(ft.freqs.sum()) > 0.0  # training: EMA advanced


def test_batch_honors_transcoder_target() -> None:
    # C: an SAETrainBatch with distinct target_sites must not be flattened to an
    # autoencoder target; an explicit y still wins.
    from saeco.data.training_data.sae_train_batch import SAETrainBatch

    tr, d = _build_gated_trainable()
    src, tgt = torch.randn(8, d), torch.randn(8, d)
    stb = SAETrainBatch(src=src, tgt=tgt, input_sites=["src"], target_sites=["tgt"])
    batch = tr._batch(stb)
    assert torch.allclose(batch.input, src)
    assert torch.allclose(batch.target, tgt)  # honors target_sites, not == input
    assert not torch.allclose(batch.input, batch.target)
    y = torch.randn(8, d)
    assert torch.allclose(tr._batch(stb, y=y).target, y)  # explicit y wins
