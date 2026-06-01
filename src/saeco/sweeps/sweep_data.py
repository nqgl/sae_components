from pathlib import Path

from paramsight.generic_restored_basemodel.generic_basemodel import GenericBaseModel

from saeco.architecture import ArchitectureBase
from saeco.architecture.arch_reload_info import ArchRef
from saeco.mlog import mlog
from saeco.sweeps import SweepableConfig


class SweepData[
    ConfigT: SweepableConfig = SweepableConfig,
    ArchT: ArchitectureBase = ArchitectureBase,
](GenericBaseModel):
    root_arch_ref: ArchRef[ConfigT, ArchT]
    sweep_id: str | None = None
    project: str | None = None

    @classmethod
    def from_arch_and_id(
        cls, arch: ArchitectureBase, sweep_id: str, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        return cls[arch_ref.cfg_t, arch_ref.arch_t](
            root_arch_ref=arch_ref,
            sweep_id=sweep_id,
            project=project,
        )

    @classmethod
    def from_arch_make_sweep(
        cls, arch: ArchitectureBase, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        sweep_id = mlog.create_sweep(arch_ref.config.to_swept_nodes, project=project)
        return cls[arch_ref.cfg_t, arch_ref.arch_t](
            root_arch_ref=arch_ref,
            sweep_id=sweep_id,
            project=project,
        )

    @classmethod
    def load(cls, path: Path) -> "SweepData":
        # GenericBaseModel reconstructs the SweepData[...] specialization from the
        # carried generic_type.
        return cls.model_validate_json(path.read_text())

    def get_sweep_number(self, name: str = "") -> int:
        proj = Path(f"./sweeprefs/{self.project}")
        num = len(list(proj.glob(f"{name}*.sweepdata")))

        while Path(f"./sweeprefs/{self.project}/{name}{num}.sweepdata").exists():
            num += 1
        return num

    def save(self, path: Path | None):
        if self.sweep_id is None:
            self.sweep_id = self.get_sweep_number()
        elif not any(c.isnumeric() for c in self.sweep_id):
            self.sweep_id += str(self.get_sweep_number(self.sweep_id))
        if path is None:
            path = Path(f"./sweeprefs/{self.project}/{self.sweep_id}.sweepdata")
            if path.exists():
                raise ValueError(f"file already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json())
        return path
