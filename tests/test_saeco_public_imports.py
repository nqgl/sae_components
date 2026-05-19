import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "saeco",
        "saeco.architecture",
        "saeco.architectures.gated",
        "saeco.architectures.vanilla",
        "saeco.components",
        "saeco.core",
        "saeco.data",
        "saeco.initializer",
        "saeco.sweeps",
        "saeco.trainer",
    ],
)
def test_public_saeco_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "symbol_name",
    [
        "Architecture",
        "ArchitectureBase",
        "InitConfig",
        "RunConfig",
        "RunSchedulingConfig",
        "SAE",
        "SweepableConfig",
        "SweepExpression",
        "SweepVar",
        "Swept",
        "TrainConfig",
        "Trainable",
        "Trainer",
        "Val",
        "do_sweep",
    ],
)
def test_saeco_top_level_reexports(symbol_name: str) -> None:
    saeco = importlib.import_module("saeco")
    assert hasattr(saeco, symbol_name)
