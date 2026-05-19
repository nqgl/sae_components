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
    num_steps: int = 50_000
    pre_bias: bool = False
    l0_target: int = 50


def show(label: str, cfg: SweepableConfig) -> None:
    bar = "-" * len(label)
    print(f"\n{label}\n{bar}")
    tree = cfg.sweep_info_tree
    print(f"  is_concrete:                       {cfg.is_concrete()}")
    print(
        f"  swept_combinations (excl. vars):   {tree._swept_combinations_count_excluding_vars()}"
    )
    print(
        f"  swept_combinations (incl. vars):   {tree.swept_combinations_count_including_vars()}"
    )
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
#    multiple fields. Each value of the var produces one run regardless
#    of how many fields it feeds into, so coupled fields move *together*
#    on one axis instead of forming a cross-product.
#
#    Example: a compute-allocation equivalence sweep. Hold the total
#    token budget fixed (512 * 2**16) while varying batch size, to check
#    whether results are invariant to how a fixed budget is allocated
#    between batch size and step count.
# ---------------------------------------------------------------------------
batch_size_weight = SweepVar(1, 2, 4, 8, 16, name="batch_size_weight")

flop_equiv = TrainCfg(
    batch_size=Val(value=512) * batch_size_weight,
    num_steps=Val(value=2**16) // batch_size_weight,
)
# 5 runs (one per var value), not 25 — batch_size and num_steps are
# locked to a single axis, and batch_size * num_steps stays ~constant.
show("3) SweepVar: fixed-budget allocation sweep", flop_equiv)


# ---------------------------------------------------------------------------
# 4) Combine: a Swept axis + a SweepVar axis -> outer product.
#    2 * 5 = 10 runs; lr varies on one axis, the budget allocation on
#    the other.
# ---------------------------------------------------------------------------
combined = TrainCfg(
    lr=Swept(1e-3, 3e-4),
    batch_size=Val(value=512) * batch_size_weight,
    num_steps=Val(value=2**16) // batch_size_weight,
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
