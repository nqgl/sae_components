import importlib
import importlib.util
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel

import wandb
from saeco.architecture.arch_reload_info import ArchClassRef
from saeco.components.model import Architecture
from saeco.misc import lazyprop
from saeco.sweeps.sweepable_config import SweepableConfig
from attrs import define, field
from saeco.mlog import mlog
import clearml


class SweepFile(Protocol):
    PROJECT: str
    cfg: SweepableConfig

    def run(self, cfg: SweepableConfig): ...


class Sweeper:
    def __init__(self, arch: Architecture):
        self.arch = arch

    @property
    def cfg(self):
        return self.arch.base_cfg

    def initialize_sweep(self):
        cfg = self.cfg
        representation = cfg.to_swept_nodes()

        sweep_id = mlog.begin_sweep(representation, project=self.sweepfile.PROJECT)
        # sweep_id = wandb.sweep(
        #     sweep=representation.to_wandb(),
        #     project=self.sweepfile.PROJECT,
        # )
        with open(self.path / "sweep_id.txt", "w") as f:
            f.write(sweep_id)

    @property
    def sweep_id(self):
        return open(self.path / "sweep_id.txt").read().strip()

    def run(self):
        mlog.init()
        basecfg: SweepableConfig = self.sweepfile.cfg

        cfg = basecfg.from_selective_sweep(dict(mlog.config))
        pod_info = dict(
            id=os.environ.get("RUNPOD_POD_ID", "local"),
            hostname=os.environ.get("RUNPOD_POD_HOSTNAME", None),
            gpu_count=os.environ.get("RUNPOD_GPU_COUNT", None),
            cpu_count=os.environ.get("RUNPOD_CPU_COUNT", None),
            public_ip=os.environ.get("RUNPOD_PUBLIC_IP", None),
            datacenter_id=os.environ.get("RUNPOD_DC_ID", None),
            volume_id=os.environ.get("RUNPOD_VOLUME_ID", None),
            cuda_version=os.environ.get("CUDA_VERSION", None),
            pytorch_version=os.environ.get("PYTORCH_VERSION", None),
        )

        wandb.config.update(
            dict(
                full_cfg=cfg.model_dump(),
                pod_info=pod_info,
            )
        )
        print(dict(wandb.config))
        self.arch.run(cfg)
        wandb.finish()

    def start_agent(self):
        wandb.agent(
            self.sweep_id,
            function=self.run,
            project=self.sweepfile.PROJECT,  # TODO change project to being from config maybe? or remove from config
        )

    def rand_run_no_agent(self):
        basecfg: SweepableConfig = self.sweepfile.cfg

        cfg = basecfg.random_sweep_configuration()
        # wandb.init()
        wandb.init(config=cfg.model_dump(), project=self.sweepfile.PROJECT)
        # wandb.config.update(cfg.model_dump())
        self.sweepfile.run(cfg)
        wandb.finish()

    def load_architecture(self, path: Path): ...


from typing import TypeVar, Generic
import json

T = TypeVar("T", bound=SweepableConfig)


class SweepData2(BaseModel, Generic[T]):
    arch_class_ref: ArchClassRef
    root_config: T
    sweep_id: str

    @classmethod
    def load(cls, path: Path) -> "SweepData":
        data = json.loads(path.read_text())
        arch_cls_ref = ArchClassRef.model_validate(data["arch_class_ref"])
        arch_cls = arch_cls_ref.get_arch_class()
        return cls[arch_cls.get_config_class()].model_validate(data)

    def save(self, path: Path | None):
        if path is None:
            path = Path(f"sweeprefs/{self.sweep_id}.json")
        path.write_text(self.model_dump_json())
        return path


class SweepData(BaseModel):
    root_arch_path: str
    sweep_id: str

    @classmethod
    def load(cls, path: Path) -> "SweepData":
        data = json.loads(path.read_text())
        return cls.model_validate(data)

    def save(self, path: Path | None):
        if path is None:
            path = Path(f"sweeprefs/{self.sweep_id}.json")
        path.write_text(self.model_dump_json())
        return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweeper for Saeco")
    parser.add_argument("arch_path", type=str)
    parser.add_argument("--sweep-id", action="store_true")
    args = parser.parse_args()
    sw = Sweeper(Architecture.load_from_path(args.arch_path), sweep_id=args.sweep_id)
    if args.init:
        sw.initialize_sweep()
    else:
        sw.start_agent()
    if args.runpod_n_instances:
        assert args.init
        from ezpod import Pods

        pods = Pods.All()
        pods.make_new_pods(args.runpod_n_instances)
        pods.sync()
        pods.setup()
        print("running!")
        pods.runpy(f"src/saeco/sweeps/sweeper.py {args.path}")
        pods.purge()


if __name__ == "__main__":
    main()
