"""Behavioral + stress tests for the sweepable DSL.

Covers the properties sweepable is supposed to guarantee:

- Swept construction forms and serialization
- is_concrete semantics (incl. nested configs and expressions)
- combination counting (excl. vs incl. sweep vars)
- select_instance_by_index enumeration: completeness, uniqueness, full
  cartesian coverage, bounds
- SweepVar coupling: one named axis shared across many fields
- SweepExpression / Val / operator algebra and typed forms
- random_sweep_configuration always yields a concrete, in-options config
- get_hash determinism and discrimination
"""

import pytest

from sweepable import SweepableConfig, SweepVar, Swept, Val


# ---------------------------------------------------------------------------
# Configs used across tests
# ---------------------------------------------------------------------------
class Leaf(SweepableConfig):
    p: int = 0


class Nested(SweepableConfig):
    leaf: Leaf
    lr: float = 1e-3


class Flat(SweepableConfig):
    a: int = 1
    b: int = 2
    c: float = 1.0


# ---------------------------------------------------------------------------
# Swept basics
# ---------------------------------------------------------------------------
class TestSwept:
    def test_variadic_equals_values_kwarg(self):
        assert Swept(1, 2, 3).values == Swept(values=[1, 2, 3]).values == [1, 2, 3]

    def test_cannot_pass_both_variadic_and_values(self):
        with pytest.raises(ValueError, match="two sources of values"):
            Swept(1, 2, values=[3, 4])

    def test_model_dump_coerces_bool_to_int(self):
        # downstream sweep backends expect ints, not bools
        assert Swept(True, False).model_dump() == {"values": [1, 0]}

    def test_single_value_swept_is_still_a_sweep(self):
        cfg = Flat(a=Swept(7))
        assert not cfg.is_concrete()
        assert cfg.sweep_info_tree.swept_combinations_count_including_vars() == 1


# ---------------------------------------------------------------------------
# is_concrete
# ---------------------------------------------------------------------------
class TestIsConcrete:
    def test_plain_config_is_concrete(self):
        assert Flat(a=1, b=2, c=3.0).is_concrete()

    def test_swept_field_makes_non_concrete(self):
        assert not Flat(a=Swept(1, 2)).is_concrete()

    def test_nested_swept_makes_root_non_concrete(self):
        assert not Nested(leaf=Leaf(p=Swept(1, 2))).is_concrete()
        assert Nested(leaf=Leaf(p=5)).is_concrete()

    def test_expression_field_is_not_concrete(self):
        w = SweepVar(1, 2, name="w")
        assert not Flat(a=Val(value=10) * w).is_concrete()


# ---------------------------------------------------------------------------
# Combination counting
# ---------------------------------------------------------------------------
class TestCombinationCounting:
    def test_concrete_config_counts_one(self):
        tree = Flat(a=1).sweep_info_tree
        assert tree.swept_combinations_count_including_vars() == 1

    def test_single_axis(self):
        tree = Flat(a=Swept(1, 2, 3)).sweep_info_tree
        assert tree.swept_combinations_count_including_vars() == 3

    def test_two_axes_are_a_product(self):
        tree = Flat(a=Swept(1, 2, 3), b=Swept(10, 20)).sweep_info_tree
        assert tree.swept_combinations_count_including_vars() == 6

    def test_nested_swept_contributes_to_product(self):
        tree = Nested(leaf=Leaf(p=Swept(1, 2, 3)), lr=Swept(0.1, 0.2)).sweep_info_tree
        assert tree.swept_combinations_count_including_vars() == 6

    def test_sweepvar_is_one_axis_regardless_of_field_count(self):
        w = SweepVar(1, 2, 4, 8, name="w")
        cfg = Flat(a=Val(value=1) * w, b=Val(value=2) * w, c=Val(value=3.0) * w)
        tree = cfg.sweep_info_tree
        # 3 fields, but one shared var => 4 runs, not 4**3
        assert tree.swept_combinations_count_including_vars() == 4
        assert tree._swept_combinations_count_excluding_vars() == 1

    def test_swept_times_sweepvar_is_a_product(self):
        w = SweepVar(1, 2, 4, name="w")
        cfg = Flat(a=Swept(10, 20), b=Val(value=100) * w)
        tree = cfg.sweep_info_tree
        assert tree.swept_combinations_count_including_vars() == 6  # 2 * 3
        assert tree._swept_combinations_count_excluding_vars() == 2


# ---------------------------------------------------------------------------
# select_instance_by_index — enumeration properties (stress)
# ---------------------------------------------------------------------------
class TestEnumeration:
    def _all_instances(self, cfg: SweepableConfig):
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        return n, [cfg.select_instance_by_index(i) for i in range(n)]

    def test_every_index_yields_concrete(self):
        cfg = Flat(a=Swept(1, 2, 3), b=Swept(10, 20))
        n, instances = self._all_instances(cfg)
        assert n == 6
        assert all(i.is_concrete() for i in instances)

    def test_full_cartesian_coverage_no_dupes_no_gaps(self):
        cfg = Flat(a=Swept(1, 2, 3), b=Swept(10, 20))
        _, instances = self._all_instances(cfg)
        seen = {(i.a, i.b) for i in instances}
        assert seen == {(a, b) for a in (1, 2, 3) for b in (10, 20)}

    def test_nested_enumeration_is_complete(self):
        cfg = Nested(leaf=Leaf(p=Swept(1, 2, 3)), lr=Swept(0.1, 0.2))
        n, instances = self._all_instances(cfg)
        seen = {(i.leaf.p, i.lr) for i in instances}
        assert n == 6
        assert seen == {(p, lr) for p in (1, 2, 3) for lr in (0.1, 0.2)}
        assert all(i.is_concrete() for i in instances)

    def test_out_of_range_raises_indexerror(self):
        cfg = Flat(a=Swept(1, 2, 3))
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        with pytest.raises(IndexError):
            cfg.select_instance_by_index(n)
        with pytest.raises(IndexError):
            cfg.select_instance_by_index(-1)

    @pytest.mark.parametrize("dims", [(2,), (2, 3), (2, 3, 4), (5, 2)])
    def test_count_matches_distinct_instances(self, dims):
        kw = {}
        names = ["a", "b"]
        # build up to 2 swept axes (Flat has a, b) sized from dims[:2]
        for name, k in zip(names, dims, strict=False):
            kw[name] = Swept(*range(k))
        cfg = Flat(**kw)
        n, instances = self._all_instances(cfg)
        expected = 1
        for k in dims[: len(names)]:
            expected *= k
        assert n == expected
        distinct = {(i.a, i.b) for i in instances}
        assert len(distinct) == expected


# ---------------------------------------------------------------------------
# SweepVar coupling — the fixed-budget allocation stress test
# ---------------------------------------------------------------------------
class BudgetCfg(SweepableConfig):
    batch_size: int = 512
    num_steps: int = 2**16


class TestSweepVarCoupling:
    def test_shared_var_keeps_product_invariant(self):
        # batch_size * num_steps should stay constant across the sweep
        w = SweepVar(1, 2, 4, 8, 16, name="batch_size_weight")
        cfg = BudgetCfg(
            batch_size=Val(value=512) * w,
            num_steps=Val(value=2**16) // w,
        )
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        assert n == 5  # one per var value, NOT 5*5

        budgets = set()
        for i in range(n):
            inst = cfg.select_instance_by_index(i)
            assert inst.is_concrete()
            budgets.add(inst.batch_size * inst.num_steps)
        assert budgets == {512 * 2**16}

    def test_var_values_each_appear_once(self):
        w = SweepVar(1, 2, 4, 8, 16, name="bw")
        cfg = BudgetCfg(batch_size=Val(value=512) * w)
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        sizes = sorted(cfg.select_instance_by_index(i).batch_size for i in range(n))
        assert sizes == [512 * k for k in (1, 2, 4, 8, 16)]


# ---------------------------------------------------------------------------
# SweepExpression / Val / operator algebra + typed forms
# ---------------------------------------------------------------------------
class ExprCfg(SweepableConfig):
    v: int = 0
    f: float = 0.0


class TestExpressionAlgebra:
    @pytest.mark.parametrize(
        ("make_expr", "expected"),
        [
            (lambda w: Val(value=10) + w, [12, 13]),
            (lambda w: Val(value=10) - w, [8, 7]),
            (lambda w: Val(value=10) * w, [20, 30]),
            (lambda w: Val(value=10) // w, [5, 3]),
            (lambda w: Val(value=2) ** w, [4, 8]),
            (lambda w: Val(value=10) % w, [0, 1]),
            (lambda w: w * Val(value=10), [20, 30]),  # __rmul__
            (lambda w: 512 * w, [1024, 1536]),  # bare int * var
        ],
    )
    def test_int_operators(self, make_expr, expected):
        w = SweepVar(2, 3, name="w")
        cfg = ExprCfg(v=make_expr(w))
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        got = [cfg.select_instance_by_index(i).v for i in range(n)]
        assert got == expected

    def test_true_division_promotes_to_float(self):
        v = SweepVar(1, 2, name="v")
        cfg = ExprCfg(f=Val(value=1.0) / v)
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        vals = [cfg.select_instance_by_index(i).f for i in range(n)]
        assert vals == [1.0, 0.5]

    def test_val_generic_type_inference(self):
        assert Val(value=5).generic_type is int
        assert Val(value=1.5).generic_type is float
        assert Val(value=True).generic_type is bool
        assert Val(value={1: 2}).generic_type == dict[int, int]

    def test_explicit_generic_forms(self):
        # the "finnicky" typed constructions: explicit params must work
        assert Val[int](value=5).generic_type is int
        lv = SweepVar[list[int]](name="lv", values=[[1, 2], [3, 4]])
        assert lv.generic_type == list[int]

    def test_sweepvar_inferred_type(self):
        assert SweepVar(1, 2, 3, name="i").generic_type is int
        assert SweepVar(1.0, 2.0, name="f").generic_type is float

    def test_expression_field_enumerates_like_its_var(self):
        w = SweepVar(1, 2, 3, name="w")
        # a SweepExpression contributes the var's axis, evaluated per run
        cfg = ExprCfg(v=Val(value=100) * w)
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        assert n == 3
        assert sorted(cfg.select_instance_by_index(i).v for i in range(n)) == [
            100,
            200,
            300,
        ]

    def test_paths_to_sweep_expressions(self):
        w = SweepVar(1, 2, name="w")
        cfg = Nested(leaf=Leaf(p=5), lr=1e-3)
        cfg2 = Nested(leaf=Leaf(p=5), lr=Val(value=1.0) * w)
        assert cfg.sweep_info_tree.get_paths_to_sweep_expressions() == []
        assert cfg2.sweep_info_tree.get_paths_to_sweep_expressions() == [["lr"]]


# ---------------------------------------------------------------------------
# random_sweep_configuration
# ---------------------------------------------------------------------------
class TestRandomSweepConfiguration:
    def test_always_concrete_and_in_options(self):
        cfg = Flat(a=Swept(5, 6, 7), b=Swept(10, 20))
        for _ in range(50):
            s = cfg.random_sweep_configuration()
            assert s.is_concrete()
            assert s.a in (5, 6, 7)
            assert s.b in (10, 20)

    def test_random_respects_sweepvar_coupling(self):
        w = SweepVar(1, 2, 4, name="w")
        cfg = BudgetCfg(
            batch_size=Val(value=512) * w,
            num_steps=Val(value=2**16) // w,
        )
        for _ in range(50):
            s = cfg.random_sweep_configuration()
            assert s.is_concrete()
            assert s.batch_size * s.num_steps == 512 * 2**16


# ---------------------------------------------------------------------------
# get_hash
# ---------------------------------------------------------------------------
class TestGetHash:
    def test_deterministic_for_equal_configs(self):
        assert Flat(a=Swept(1, 2, 3)).get_hash() == Flat(a=Swept(1, 2, 3)).get_hash()

    def test_discriminates_different_configs(self):
        assert Flat(a=Swept(1, 2, 3)).get_hash() != Flat(a=Swept(1, 2, 4)).get_hash()
        assert Flat(a=1).get_hash() != Flat(a=2).get_hash()

    def test_distinct_enumerated_instances_have_distinct_hashes(self):
        cfg = Flat(a=Swept(1, 2, 3), b=Swept(10, 20))
        n = cfg.sweep_info_tree.swept_combinations_count_including_vars()
        hashes = {cfg.select_instance_by_index(i).get_hash() for i in range(n)}
        assert len(hashes) == n
