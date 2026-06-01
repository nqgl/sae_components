from pathlib import Path
from typing import Any, ClassVar

from paramsight import TypeVarValue
from paramsight.generic_restored_basemodel.generic_basemodel import GenericBaseModel

from saeco.architecture.architecture import ArchitectureBase
from saeco.sweeps import SweepableConfig


class ArchRef[ConfigT: SweepableConfig, ArchT: ArchitectureBase](GenericBaseModel):
    """Serializable reference to an architecture + its run config.

    A :class:`GenericBaseModel`, so the type arguments — the run-config type
    ``ConfigT`` and the architecture class ``ArchT`` — round-trip through JSON
    automatically (no hand-rolled class-ref / re-derivation). ``cfg_t`` / ``arch_t``
    recover them from a (de)serialized instance.
    """

    config: ConfigT
    cfg_t: ClassVar = TypeVarValue[ConfigT]()
    arch_t: ClassVar = TypeVarValue[ArchT]()

    @classmethod
    def open(cls, path: Path) -> "ArchRef[Any, Any]":
        # GenericBaseModel reconstructs the right ArchRef[...] specialization from
        # the carried generic_type; no need to pre-specialize when opening.
        return cls.model_validate_json(path.read_text())

    @classmethod
    def from_arch(cls, arch: "ArchitectureBase[Any]") -> "ArchRef[Any, Any]":
        return cls[type(arch.run_cfg), type(arch)](config=arch.run_cfg)

    def load_arch(
        self,
        state_dict=None,
        device="cuda",
        xcls: "type[ArchitectureBase[Any]] | None" = None,
    ) -> "ArchitectureBase[Any]":
        arch_cls: type[ArchitectureBase[Any]] = self.arch_t
        if xcls is not None:
            if arch_cls in xcls.mro():
                arch_cls = xcls
            elif issubclass(xcls, arch_cls):
                arch_cls = xcls
            else:
                raise ValueError(f"xcls {xcls} is not a subclass of {arch_cls}")
        return arch_cls(self.config, state_dict=state_dict, device=device)

    def save(self, path: Path):
        path.write_text(self.model_dump_json())
