"""Regression tests pinning Architecture config-class resolution.

These lock in the resolved config types (``RunConfigT`` and ``ArchConfigT``) so
the switch from the ``get_config_class()`` / ``get_arch_config_class()`` methods
to ``paramsight.TypeVarValue`` descriptors (``config_class`` / ``arch_config_class``)
is provably behavior-preserving — same resolved types before and after, and the
bare unspecialized base still refuses to resolve.
"""

import pytest

from saeco.architecture.architecture import ArchitectureBase
from saeco.architectures.gated.arch import Gated, GatedConfig
from saeco.architectures.vanilla.vanilla_model import VanillaConfig, VanillaSAE
from saeco.trainer.run_config import RunConfig


def test_vanilla_config_classes():
    assert VanillaSAE.arch_config_class is VanillaConfig
    assert VanillaSAE.config_class == RunConfig[VanillaConfig]


def test_gated_config_classes():
    assert Gated.arch_config_class is GatedConfig
    assert Gated.config_class == RunConfig[GatedConfig]


def test_bare_base_does_not_resolve():
    # An unspecialized ArchitectureBase has no concrete RunConfigT (and no
    # typevar default), so resolving its config class must fail rather than
    # silently return a placeholder.
    with pytest.raises((ValueError, LookupError)):
        _ = ArchitectureBase.config_class
