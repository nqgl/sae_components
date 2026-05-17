import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "saeco_research",
        "saeco_research.architectures",
        "saeco_research.evaluation.filtered",
    ],
)
def test_research_modules_import_without_eager_side_effects(
    module_name: str,
) -> None:
    importlib.import_module(module_name)
