import importlib
import importlib.util
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Protocol

import wandb
from saeco.components.model import Architecture
from saeco.misc import lazyprop
from saeco.sweeps.sweepable_config import SweepableConfig
from attrs import define, field
from saeco.mlog import mlog


class SweepFile(Protocol):
    PROJECT: str
    cfg: SweepableConfig

    def run(self, cfg: SweepableConfig): ...


class Sweeper:
    def __init__(self, arch: Architecture):
        self.arch = arch

    # @cached_property
    # def sweepfile(self) -> SweepFile:
    #     spec = importlib.util.spec_from_file_location(
    #         self.full_name, str(self.path / f"{self.module_name}.py")
    #     )
    #     sweepfile = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(sweepfile)
    #     return sweepfile

    @property
    def cfg(self):
        return self.arch.base_cfg

    def initialize_sweep(self):
        cfg = self.cfg
        representation = cfg.to_swept_nodes()

        sweep_id = mlog.create_sweep(representation, project=self.sweepfile.PROJECT)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweeper for Saeco")
    parser.add_argument("path", type=str)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--runpod-n-instances", type=int)
    parser.add_argument("--module-name", type=str, default="sweepfile")
    args = parser.parse_args()
    sw = Sweeper(args.path, module_name=args.module_name)
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
