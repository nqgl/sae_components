"""
The sweep config DSL
====================

Sweeping is a first-class feature of `SweepableConfig`. You don't write a
separate sweep file: you put `Swept(...)`, `SweepVar(...)`, or
`SweepExpression(...)` directly inside your config, and the runner
enumerates the resulting concrete configurations.

This example doesn't touch a model or training — it's pure config DSL.
Run it: `python examples/sweep_dsl.py`

Note on type checkers: pyright reports `Swept[T]` as not assignable to
`T`. SweepableConfig handles this widening at runtime; the type checker
doesn't see it. The examples below run correctly.
"""

from saeco import SweepableConfig, SweepVar, Swept, Val


class TrainCfg(SweepableConfig):
    lr: float = 1e-3
    batch_size: int = 4096
    pre_bias: bool = False
    l0_target: int = 50


def show(label: str, cfg: SweepableConfig) -> None:
    bar = "-" * len(label)
    print(f"\n{label}\n{bar}")
    tree = cfg.sweep_info_tree
    print(f"  is_concrete:                       {cfg.is_concrete()}")
    print(f"  swept_combinations (excl. vars):   {tree._swept_combinations_count_excluding_vars()}")
    print(f"  swept_combinations (incl. vars):   {tree.swept_combinations_count_including_vars()}")
    if not cfg.is_concrete():
        sample = cfg.random_sweep_configuration()
        print(f"  random concrete sample:            {sample.model_dump()}")


# ---------------------------------------------------------------------------
# 1) A concrete config has no Swept fields and represents a single run.
# ---------------------------------------------------------------------------
concrete = TrainCfg(lr=1e-3, batch_size=4096, pre_bias=False, l0_target=50)
show("1) Concrete config (no sweep)", concrete)


# ---------------------------------------------------------------------------
# 2) A `Swept` field becomes one axis of a grid.
#    Two Swept fields -> outer product (2 * 3 = 6 runs).
# ---------------------------------------------------------------------------
gridded = TrainCfg(
    lr=Swept(1e-3, 3e-4, 1e-4),
    pre_bias=Swept(True, False),
)
show("2) Gridded with two Swept axes", gridded)


# ---------------------------------------------------------------------------
# 3) A `SweepVar` is a *named* sweep axis that can be referenced from
#    multiple fields. Each value of the var produces one run, regardless
#    of how many fields the var feeds into.
# ---------------------------------------------------------------------------
batch_size_mult = SweepVar(1, 2, 4, name="batch_size_mult")

with_var = TrainCfg(
    lr=1e-3,
    # The same SweepVar feeds two fields. The grid still has 3 runs
    # (not 9), because it's one axis.
    batch_size=batch_size_mult * 4096,
    l0_target=batch_size_mult * 25,  # vary l0_target proportionally
)
show("3) Single SweepVar feeding two fields", with_var)


# ---------------------------------------------------------------------------
# 4) Combine: a Swept axis + a SweepVar axis -> outer product.
#    2 * 3 = 6 runs; lr varies on one axis, batch_size_mult on the other.
# ---------------------------------------------------------------------------
combined = TrainCfg(
    lr=Swept(1e-3, 3e-4),
    batch_size=batch_size_mult * 4096,
    l0_target=batch_size_mult * 25,
)
show("4) Combined Swept + SweepVar", combined)


# ---------------------------------------------------------------------------
# 5) `Val` lets you wrap a constant so it can participate in sweep
#    expressions (Val/SweepVar arithmetic). Useful when the constant's
#    type isn't an int/float.
# ---------------------------------------------------------------------------
expr_var = SweepVar(1, 2, 4, name="expr_var")
with_expr = TrainCfg(
    batch_size=Val(value=4096) * expr_var,
    # Use integer division (//) — a Swept[int] field needs an int result.
    l0_target=Val(value=100) // expr_var,
)
show("5) Val-based sweep expression", with_expr)
