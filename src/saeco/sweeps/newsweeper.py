from contextlib import contextmanager
import importlib
import importlib.util
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Protocol, Generator
from pydantic import BaseModel
import time
import wandb
from saeco.architecture.arch_reload_info import ArchClassRef, ArchRef
from saeco.misc import lazyprop
from saeco.sweeps.sweepable_config import SweepableConfig
from attrs import define, field
from saeco.mlog import mlog
from saeco.architecture import Architecture
from .SweepRunner import SweepRunner
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ezpod import Pods


class SweepFile(Protocol):
    PROJECT: str
    cfg: SweepableConfig

    def run(self, cfg: SweepableConfig): ...


from typing import TypeVar, Generic
import json

T = TypeVar("T", bound=SweepableConfig)


# class SweepData2(BaseModel, Generic[T]):
#     arch_class_ref: ArchClassRef
#     root_config: T
#     sweep_id: str

#     @classmethod
#     def load(cls, path: Path) -> "SweepData":
#         data = json.loads(path.read_text())
#         arch_cls_ref = ArchClassRef.model_validate(data["arch_class_ref"])
#         arch_cls = arch_cls_ref.get_arch_class()
#         return cls[arch_cls.get_config_class()].model_validate(data)

#     def save(self, path: Path | None):
#         if path is None:
#             path = Path(f"sweeprefs/{self.sweep_id}.json")
#         path.write_text(self.model_dump_json())
#         return path


class SweepData(BaseModel, Generic[T]):
    root_arch_ref: ArchRef[T]
    sweep_id: str | None = None
    project: str | None = None

    @classmethod
    def from_arch_and_id(
        cls, arch: Architecture, sweep_id: str, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        return cls[arch_ref.class_ref.get_arch_class().get_config_class()](
            root_arch_ref=arch_ref.model_dump(),
            sweep_id=sweep_id,
            project=project,
        )

    @classmethod
    def from_arch_make_sweep(
        cls, arch: Architecture, project: str | None = None
    ) -> "SweepData":
        arch_ref = ArchRef.from_arch(arch)
        sweep_id = mlog.create_sweep(arch_ref.config.to_swept_nodes(), project=project)
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

    def save(self, path: Path | None):
        if path is None:
            path = Path(f"./sweeprefs/{self.project}/{self.sweep_id}.sweepdata")
            if path.exists():
                raise ValueError(f"file already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json())
        return path


import uuid


@define
class SweepManager:
    arch: Architecture
    sweep_data: SweepData | None = field(default=None)
    sweep_data_path: Path | None = field(default=None)
    ezpod_group: str | None = field(default=None)

    @property
    def cfg(self):
        return self.arch.run_cfg

    def initialize_sweep(self, project=None, custom_sweep=False):
        assert self.sweep_data is None and self.sweep_data_path is None
        if self.cfg.is_concrete():
            raise ValueError(
                "tried to initialize sweep on a config with no swept fields"
            )
        if custom_sweep:
            import datetime

            sweep_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            sweep_id = mlog.create_sweep(self.cfg.to_swept_nodes(), project=project)
        self.sweep_data = SweepData.from_arch_and_id(
            self.arch, sweep_id, project=project
        )
        self.sweep_data_path = self.sweep_data.save(None)

    def rand_run_no_agent(self):
        arch_ref = ArchRef.from_arch(self.arch)
        sweeprunner = SweepRunner(arch_ref)
        return sweeprunner.run_random_instance()

    def local_sweep(self):
        arch_ref = ArchRef.from_arch(self.arch)
        sweeprunner = SweepRunner(arch_ref, self.sweep_data.sweep_id)
        return sweeprunner.start_sweep_agent()

    def local_custom_sweep(self):
        # root = self.sweep_data.root_arch_ref.config
        root = self.arch.run_cfg
        root_swept = root.to_swept_nodes()
        N = root_swept.swept_combinations_count_including_vars()
        for i in range(N):
            cfg = root.from_selective_sweep(root_swept.select_instance_by_index(i))
            cfg_hash = cfg.get_hash()
            runner = SweepRunner.from_sweepdata(
                self.sweep_data, sweep_index=i, sweep_hash=cfg_hash
            )
            runner.start_sweep_agent()

    def load_architecture(self, path: Path): ...

    def get_worker_run_command(self, extra_args: str = ""):
        path = (
            self.sweep_data_path
            if not self.sweep_data_path.is_absolute()
            else self.sweep_data_path.relative_to(Path.cwd())
        )
        return f"src/saeco/sweeps/sweeprunner_cli.py {path} {extra_args}"

    def get_worker_run_commands_for_manual_sweep(self):
        root = self.arch.run_cfg
        root_swept = root.to_swept_nodes()
        N = root_swept.swept_combinations_count_including_vars()
        variants = []
        for i in range(N):
            cfg = root.from_selective_sweep(root_swept.select_instance_by_index(i))
            variants.append((i, cfg.get_hash()))
        return [
            self.get_worker_run_command(f"--sweep-index {i} --sweep-hash {h}")
            for i, h in variants
        ]

    def run_sweep_on_pods(
        self,
        new_pods=None,
        purge_after=True,
        challenge_file="src/saeco/sweeps/challenge.py",
        keep_after=False,
    ):
        with self.created_pods(new_pods, keep=keep_after) as pods:
            pods.runpy(
                self.get_worker_run_command(),
                purge_after=purge_after,
                challenge_file=challenge_file,
            )

    def run_sweep_on_pods_with_monitoring(
        self,
        new_pods=None,
        purge_after=True,
        challenge_file=None,
        # challenge_file="src/saeco/sweeps/challenge.py",
        keep_after=False,
        setup_min=None,
    ):

        with self.created_pods(new_pods, keep=keep_after, setup_min=setup_min) as pods:
            print("running on remotes")
            task = pods.runpy_with_monitor(
                self.get_worker_run_command(),
                purge_after=purge_after,
                challenge_file=challenge_file,
            )

    def run_manual_sweep_with_monitoring(
        self,
        new_pods=None,
        purge_after=True,
        keep_after=False,
        setup_min=None,
        prefix_vars=None,
    ):

        with self.created_pods(new_pods, keep=keep_after, setup_min=setup_min) as pods:
            print("running on remotes")
            task = pods.runpy_with_monitor(
                self.get_worker_run_commands_for_manual_sweep(),
                purge_after=purge_after,
                challenge_file=None,
                prefix_vars=prefix_vars,
            )

    @contextmanager
    def created_pods(
        self,
        num_pods=None,
        keep=False,
        setup_min=None,
    ) -> Generator["Pods", None, None]:
        from ezpod import Pods

        if num_pods is None:
            num_pods = int(input("Enter number of pods: "))
        pods = Pods.All(group=self.ezpod_group)
        pods.make_new_pods(num_pods)
        pods.sync()
        pods.setup()
        pods.EZPOD_MIN_COMPLETE_TO_CONTINUE = setup_min
        try:
            yield pods
        finally:
            if not keep:
                try:
                    pods.purge()
                except Exception as e:
                    try:
                        print("pod purge failed, retrying")
                        time.sleep(10)
                        Pods.All(group=self.ezpod_group).purge()
                    except Exception as e:
                        print(
                            "failed twice to purge subset of pods, purging all pods in 10 seconds"
                        )
                        time.sleep(10)
                        Pods.All().purge()
                        raise e
                    raise e
