"""Regression tests pinning ArchRef / SweepData / Architecture.load serialization.

These lock the JSON round-trip behavior so the switch of ``ArchRef``/``SweepData``
to ``paramsight.GenericBaseModel`` (with ``ArchT`` as a generic parameter,
replacing ``ArchClassRef``, and ``from_json`` removed) is provably
behavior-preserving: the reconstructed config keeps its exact (nested-)generic
type ``RunConfig[VanillaConfig]`` and equals the original, and ``load`` still
reconstructs the right Architecture class (the ``xcls`` override capability).
"""

from saeco import InitConfig, RunConfig, TrainConfig
from saeco.architecture.arch_reload_info import ArchRef
from saeco.architectures.vanilla import VanillaConfig, VanillaSAE
from saeco.data.config.data_config_definitions import (
    GEMMA_4_DEFAULT_D_DATA,
    gemma_4_lmsys_chat,
)
from saeco.sweeps.sweep_data import SweepData


def _make_arch() -> VanillaSAE:
    # A real-but-lazy config (architectures lazy-load; constructing this does
    # not load a model).
    cfg = RunConfig[VanillaConfig](
        train_cfg=TrainConfig(data_cfg=gemma_4_lmsys_chat(14)),
        init_cfg=InitConfig(d_data=GEMMA_4_DEFAULT_D_DATA, dict_mult=4),
        arch_cfg=VanillaConfig(),
    )
    return VanillaSAE(cfg)


def test_archref_roundtrip_preserves_config_type():
    arch = _make_arch()
    ref = ArchRef.from_arch(arch)
    back = ArchRef.model_validate_json(ref.model_dump_json())
    assert back.config == ref.config
    assert back.config.__class__ is RunConfig[VanillaConfig]
    # ArchT round-trips: the descriptors recover both type args.
    assert back.arch_t is VanillaSAE
    assert back.cfg_t is RunConfig[VanillaConfig]


def test_sweepdata_roundtrip_preserves_config_type():
    arch = _make_arch()
    sd = SweepData.from_arch_and_id(arch, sweep_id="s1", project="p")
    back = SweepData.model_validate_json(sd.model_dump_json())
    assert back.sweep_id == "s1"
    assert back.root_arch_ref.config == sd.root_arch_ref.config
    assert back.root_arch_ref.config.__class__ is RunConfig[VanillaConfig]


def test_architecture_load_reconstructs_arch_and_config(tmp_path):
    arch = _make_arch()
    arch.save_to_path(tmp_path / "arch", save_weights=False)
    loaded = VanillaSAE.load(tmp_path / "arch", load_weights=False)
    assert type(loaded) is VanillaSAE
    assert loaded.run_cfg == arch.run_cfg
    assert loaded.run_cfg.__class__ is RunConfig[VanillaConfig]
