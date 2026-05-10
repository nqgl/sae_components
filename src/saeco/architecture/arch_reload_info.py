import importlib
import inspect
import json
from pathlib import Path
from types import get_original_bases
from typing import TYPE_CHECKING, Any, Self

from paramsight.generic_restored_basemodel.typeref import ObjRef
from pydantic import BaseModel, Field

from saeco.architecture.architecture import ArchitectureBase
from saeco.sweeps import SweepableConfig


def get_src(obj):
    if obj.__module__ == "builtins":
        return "builtins"
    module = importlib.import_module(obj.__module__)
    return inspect.getsource(module)


class ArchClassRef(ObjRef[type[ArchitectureBase]]):
    @classmethod
    def from_arch(cls, arch: "ArchitectureBase[Any]") -> "ArchClassRef":
        return cls.from_obj(arch.__class__)

    def get_arch_class(self, assert_unchanged: bool = False):
        return self.get_obj(strict=assert_unchanged)


class ArchRef[T: SweepableConfig](BaseModel):
    class_ref: ArchClassRef
    config: T = Field()

    @classmethod
    def open(
        cls, path: Path, xcls: "type[ArchitectureBase[Any]] | None" = None
    ) -> Self:
        try:
            bases = get_original_bases(cls)
            if hasattr(bases[0], "__args__") and bases[0].__args__[0] is not T:
                raise ValueError(
                    f"ArchRef {cls} should not be specialized when opening "
                    f"a stored archref\n"
                    f"in the future, we may implement validation to the specialized "
                    f"value when loading an ArchRef"
                )
        except AttributeError:
            pass
        return cls.from_json(json.loads(path.read_text()), xcls=xcls)

    @classmethod
    def from_json(cls, d: dict, xcls: "type[ArchitectureBase[Any]] | None" = None):
        arch = ArchClassRef.model_validate(d["class_ref"]).get_arch_class()

        if xcls is not None and arch in xcls.mro():
            arch = xcls

        config_class = arch.get_config_class()
        archref_cls = cls[config_class]
        return archref_cls.model_validate(d)

    def load_arch(
        self,
        state_dict=None,
        device="cuda",
        xcls: "type[ArchitectureBase[Any]] | None" = None,
    ):
        arch_cls = self.class_ref.get_arch_class()
        if xcls is not None:
            if arch_cls in xcls.mro():
                arch_cls = xcls
            elif issubclass(xcls, arch_cls):
                arch_cls = xcls
            else:
                raise ValueError(f"xcls {xcls} is not a subclass of {arch_cls}")
        return arch_cls(self.config, state_dict=state_dict, device=device)

    @classmethod
    def from_arch(cls, arch: "ArchitectureBase[T]") -> "ArchRef[T]":
        return cls(
            class_ref=ArchClassRef.from_arch(arch),
            config=arch.run_cfg,
        )

    def save(self, path: Path):
        path.write_text(self.model_dump_json())
