from pydantic import BaseModel, Field

import torch
import inspect
import importlib
from pathlib import Path
import json
from typing import TYPE_CHECKING, Generic, TypeVar


from saeco.trainer.run_config import RunConfig

if TYPE_CHECKING:
    from .architecture import Architecture
from saeco.sweeps import SweepableConfig

MODEL_WEIGHTS_PATH_EXT = ".weights.safetensors"
AVERAGED_WEIGHTS_PATH_EXT = ".avg_weights.safetensors"
ARCH_REF_PATH_EXT = ".arch_ref"


def get_src(obj):
    module = importlib.import_module(obj.__module__)
    return inspect.getsource(module)


class ArchClassRef(BaseModel):
    module: str
    cls_name: str
    source_backup: str = None

    @classmethod
    def from_arch(cls, arch: "Architecture") -> "ArchClassRef":
        return cls(
            module=arch.__class__.__module__,
            cls_name=arch.__class__.__name__,
            source_backup=get_src(arch.__class__),
        )

    def get_arch_class(self, assert_unchanged: bool = False):
        module = importlib.import_module(self.module)
        arch_cls = getattr(module, self.cls_name)
        from .architecture import Architecture

        assert issubclass(arch_cls, Architecture)
        if get_src(arch_cls) != self.source_backup:
            print(
                """
                warning: loaded architecture source code appears to have changed since the model was saved. 
                This may cause issues.
                (but not necessarily)
                """
            )
            if assert_unchanged:
                raise ValueError(
                    "loaded architecture source code has changed since this model was saved"
                )
        return arch_cls


T = TypeVar("T", bound=SweepableConfig)


T2 = TypeVar("T2", bound=SweepableConfig)


class ArchRef(BaseModel, Generic[T]):
    class_ref: ArchClassRef
    config: T = Field()

    @classmethod
    def open(cls, path: Path) -> "ArchRef":
        if (
            hasattr(cls.__orig_bases__[0], "__args__")
            and cls.__orig_bases__[0].__args__[0] is not T
        ):
            raise ValueError("generic type T must not be instantiated")
        return cls.from_json(json.loads(path.read_text()))

    @classmethod
    def from_json(cls, d):

        config_class = (
            ArchClassRef.model_validate(d["class_ref"])
            .get_arch_class()
            .get_config_class()
        )
        archref_cls = cls[config_class]
        return archref_cls.model_validate(d)

    def load_arch(self, state_dict=None, device="cuda"):
        arch_cls = self.class_ref.get_arch_class()
        return arch_cls(self.config, state_dict=state_dict, device=device)

    @classmethod
    def from_arch(cls, arch: "Architecture") -> "ArchRef":
        return cls(
            class_ref=ArchClassRef.from_arch(arch),
            config=arch.run_cfg,
        )

    def save(self, path: Path):
        path.write_text(self.model_dump_json())


class ArchStoragePaths(BaseModel):
    path: Path

    # @property
    # def cfg(self):
    #     return self.path.with_suffix(CFG_PATH_EXT)

    # @property
    # def arch_cls_ref(self):
    #     return self.path.with_suffix(ARCH_CLASS_REF_PATH_EXT)

    @property
    def arch_ref(self):
        return self.stempath.with_suffix(ARCH_REF_PATH_EXT)

    @property
    def model_weights(self):
        return self.stempath.with_suffix(MODEL_WEIGHTS_PATH_EXT)

    @property
    def averaged_weights(self):
        return self.stempath.with_suffix(AVERAGED_WEIGHTS_PATH_EXT)

    @classmethod
    def from_path(cls, path: Path):
        if isinstance(path, cls):
            return path
        return cls(path=path)

    def load_arch(
        self, load_weights=None, averaged_weights=False, device="cuda", state_dict=None
    ):
        from .arch_reload_info import ArchClassRef, ArchRef

        assert load_weights or not averaged_weights
        if state_dict is not None and (load_weights or averaged_weights):
            raise ValueError(
                "state_dict cannot be set if load_weights or averaged_weights is set"
            )
        if self.model_weights.exists():
            if load_weights is None:
                raise ValueError(
                    f"weights exist at {self.model_weights}, but load_weights is not set"
                )
            if load_weights:
                state_dict = (
                    torch.load(self.averaged_weights)
                    if averaged_weights
                    else torch.load(self.model_weights)
                )
        else:
            if load_weights:
                raise ValueError(
                    f"weights do not exist at {self.model_weights}, but load_weights is set"
                )
        arch_ref = ArchRef.open(self.arch_ref)
        arch_inst = arch_ref.load_arch(state_dict=state_dict, device=device)
        return arch_inst

    def exists(self):
        return (
            self.arch_ref.exists()
            or self.model_weights.exists()
            or self.averaged_weights.exists()
        )

    @property
    def stempath(self):
        return self.path.with_name(self.path.stem.split(".")[0])
