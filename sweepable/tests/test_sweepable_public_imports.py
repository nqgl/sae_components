import importlib

import pytest


@pytest.mark.parametrize(
    "symbol_name",
    [
        "SweepableConfig",
        "SweepExpression",
        "SweepVar",
        "Swept",
        "Val",
    ],
)
def test_sweepable_top_level_reexports(symbol_name: str) -> None:
    sweepable = importlib.import_module("sweepable")
    assert hasattr(sweepable, symbol_name)
