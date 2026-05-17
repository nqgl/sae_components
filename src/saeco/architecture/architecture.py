import typing
from abc import abstractmethod
from pathlib import Path
from typing import Any, overload

import torch
from paramsight import get_resolved_typevars_for_base, takes_alias

from saeco.architecture.arch_storage_paths import ArchStoragePaths
from saeco.sweeps import SweepableConfig


class ArchitectureBase[RunConfigT: SweepableConfig = SweepableConfig]:
    """Base architecture protocol for sweepable run configs.

    This class intentionally owns only config instantiation, reload/save metadata,
    and sweep orchestration. Training/model semantics belong in subclasses.

    special methods to override:
    - setup()
    - _save_weights_by_default()
    - _state_dict_for_save()
    - run_training()

    """

    def __init__(
        self,
        run_cfg: RunConfigT,
        state_dict: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
    ):
        self.run_cfg: RunConfigT = run_cfg
        self.state_dict: dict[str, Any] | None = state_dict
        self._instantiated: bool = False
        self._setup_complete: bool = False
        self.device: torch.device | str = device

    def instantiate(self, inst_cfg: dict[str, Any] | None = None):
        if inst_cfg:
            self.run_cfg = self.run_cfg.from_selective_sweep(inst_cfg)
        assert self.run_cfg.is_concrete()
        self._instantiated = True
        self._setup()

    def _setup(self):
        assert self._instantiated
        self.setup()
        self._setup_complete = True

    @abstractmethod
    def setup(self) -> None: ...

    @takes_alias
    @classmethod
    def get_config_class(cls) -> type[RunConfigT]:
        if cls is ArchitectureBase:
            raise ValueError(
                "ArchitectureBase class must not be generic to get config class"
            )
        try:
            (config_class,) = get_resolved_typevars_for_base(cls, ArchitectureBase)
        except Exception as e:
            raise ValueError("Failed in config class lookup") from e
        if isinstance(config_class, typing.TypeVar):
            raise ValueError(
                "ArchitectureBase class must not be generic to get config class"
            )
        return config_class

    def _save_weights_by_default(self) -> bool:
        return False

    def _state_dict_for_save(self) -> dict[str, Any]:
        raise ValueError(f"{type(self).__name__} does not support saving weights")

    def save_to_path(
        self,
        path: Path | ArchStoragePaths,
        save_weights: bool | None = None,
        averaged_weights: bool | None = None,
    ):
        if isinstance(path, Path):
            path = ArchStoragePaths.from_path(path)
        path.path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            self.save_to_path(
                path.path.with_name(f"{path.path.name}_1"),
                save_weights=save_weights,
                averaged_weights=averaged_weights,
            )
            raise ValueError(
                f"file already existed at {path}, wrote to {path.path.name}_1"
            )

        from .arch_reload_info import ArchRef

        arch_ref = ArchRef.from_arch(self)

        path.arch_ref.write_text(arch_ref.model_dump_json())
        should_save_weights = (
            self._save_weights_by_default() if save_weights is None else save_weights
        )
        if should_save_weights:
            torch.save(self._state_dict_for_save(), path.model_weights)  # type: ignore
        if averaged_weights is not None:
            torch.save(averaged_weights, path.averaged_weights)  # type: ignore
        return path

    @classmethod
    def load(
        cls,
        path: Path | ArchStoragePaths,
        load_weights: bool | None = None,
        averaged_weights: bool | None = False,
    ) -> "ArchitectureBase":
        return ArchStoragePaths.from_path(path).load_arch(
            load_weights=load_weights, averaged_weights=averaged_weights, xcls=cls
        )

    @abstractmethod
    def run_training(self): ...

    def get_sweep_manager(self, ezpod_group=None):
        from saeco.sweeps.newsweeper import SweepManager

        return SweepManager(self, ezpod_group=ezpod_group)

    def save_sweepref_and_get_py_commands(
        self,
        project: str,
        gpus_per_run: int,
        clivars: str = (
            'TORCH_LOGS="graph_breaks,recompiles"  '
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        ),
        pre_commands: str = "",
        pyname: str | None = None,
    ) -> list[str]:
        sm = self.get_sweep_manager()
        sm.initialize_sweep(project=project)

        commands = (
            sm.get_worker_run_commands_for_manual_sweep(suffix="--distributed-skip-log")
            if gpus_per_run > 1
            else sm.get_worker_run_commands_for_manual_sweep()
        )
        return to_py_cmd(
            commands,
            pyname=pyname
            or ("python3" if gpus_per_run == 1 else f"composer -n {gpus_per_run}"),
            challenge_file=None,
            prefix_vars=pre_commands + clivars,
        )


@overload
def to_py_cmd(
    cmd: str,
    pyname: str,
    challenge_file: str | None = None,
    prefix_vars: str | None = None,
) -> str: ...
@overload
def to_py_cmd(
    cmd: list[str],
    pyname: str,
    challenge_file: str | None = None,
    prefix_vars: str | None = None,
) -> list[str]: ...
def to_py_cmd(  # Semi temporary code duplication of something that also exists in ezpod
    cmd: str | list[str],
    pyname: str,
    challenge_file: str | None = None,
    prefix_vars: str | None = None,
) -> str | list[str]:
    if isinstance(cmd, list):
        return [
            to_py_cmd(
                c, pyname=pyname, challenge_file=challenge_file, prefix_vars=prefix_vars
            )
            for c in cmd
        ]
    cmd = f"{pyname} {cmd}"
    if prefix_vars:
        cmd = f"{prefix_vars} {cmd}"
    if challenge_file:
        cmd = f"{pyname} {challenge_file}; {cmd}"
    return cmd
