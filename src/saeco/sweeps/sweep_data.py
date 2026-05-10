import json
from pathlib import Path
from typing import Generic

from paramsight import takes_alias
from pydantic import BaseModel

from saeco.architecture import ArchitectureBase
from saeco.architecture.arch_reload_info import ArchRef
from saeco.mlog import mlog
from saeco.sweeps import SweepableConfig


class SweepData[T: SweepableConfig = SweepableConfig](BaseModel):
    root_arch_ref: ArchRef[T]
    sweep_id: str | None = None
    project: str | None = None

    @classmethod
    def from_arch_and_id(
        cls, arch: ArchitectureBase, sweep_id: str, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        # arch_ref.class_ref.get_arch_class().get_arch_config_class()
        return cls[arch_ref.class_ref.get_arch_class().get_config_class()](
            root_arch_ref=arch_ref.model_dump(),
            sweep_id=sweep_id,
            project=project,
        )

    @classmethod
    def from_arch_make_sweep(
        cls, arch: ArchitectureBase, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        sweep_id = mlog.create_sweep(arch_ref.config.to_swept_nodes, project=project)
        return cls[arch_ref.class_ref.get_arch_class().get_config_class()](
            root_arch_ref=arch_ref.model_dump(),
            sweep_id=sweep_id,
            project=project,
        )

    @classmethod
    def load(cls, path: Path) -> "SweepData":
        data = json.loads(path.read_text())
        arch_json = data["root_arch_ref"]
        arch_ref = ArchRef.from_json(arch_json)
        return cls[
            arch_ref.class_ref.get_arch_class().get_config_class()
        ].model_validate_json(path.read_text())

    def get_sweep_number(self, name: str = "") -> int:
        proj = Path(f"./sweeprefs/{self.project}")
        num = len(list(proj.glob(f"{name}*.sweepdata")))

        while Path(f"./sweeprefs/{self.project}/{name}{num}.sweepdata").exists():
            num += 1
        return num

    def save(self, path: Path | None):
        if self.sweep_id is None:
            self.sweep_id = self.get_sweep_number()
        elif not any([c.isnumeric() for c in self.sweep_id]):
            self.sweep_id += str(self.get_sweep_number(self.sweep_id))
        if path is None:
            path = Path(f"./sweeprefs/{self.project}/{self.sweep_id}.sweepdata")
            if path.exists():
                raise ValueError(f"file already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json())
        return path
