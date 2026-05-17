from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from saeco.architecture.architecture import ArchitectureBase
MODEL_WEIGHTS_PATH_EXT = ".weights.safetensors"
AVERAGED_WEIGHTS_PATH_EXT = ".avg_weights.safetensors"
ARCH_REF_PATH_EXT = ".arch_ref"


class ArchStoragePaths(BaseModel):
    path: Path

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
    def from_path(cls, path: Path | Self):
        if isinstance(path, cls):
            return path
        return cls(path=path)

    def load_arch(
        self,
        load_weights: bool | None = None,
        averaged_weights: bool | None = False,
        device: str | torch.device = "cuda",
        state_dict: dict[str, Any] | None = None,
        xcls=None,
    ) -> "ArchitectureBase[Any]":
        from .arch_reload_info import ArchRef

        assert load_weights or not averaged_weights
        if state_dict is not None and (load_weights or averaged_weights):
            raise ValueError(
                "state_dict cannot be set if load_weights or averaged_weights is set"
            )
        if self.model_weights.exists():
            if load_weights is None:
                raise ValueError(
                    f"weights exist at {self.model_weights}, but load_weights is not "
                    "set"
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
                    f"weights do not exist at {self.model_weights}, "
                    "but load_weights is set"
                )
        arch_ref = ArchRef.open(self.arch_ref, xcls=xcls)
        arch_inst = arch_ref.load_arch(state_dict=state_dict, device=device, xcls=xcls)
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
