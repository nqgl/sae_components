import importlib
import importlib.util
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Protocol

import wandb
from saeco.sweeps.sweepable_config import SweepableConfig


class SweepFile(Protocol):
    PROJECT: str
    cfg: SweepableConfig

    def run(self, cfg: SweepableConfig): ...


class Sweeper:
    def __init__(self, path, module_name="sweepfile"):
        module_name = module_name.replace(".py", "")
        self.path = Path(path)
        pkg = str(self.path).split("src/")[-1].replace("/", ".")
        self.module_name = module_name
        self.full_name = f"{pkg}.{module_name}"

    @cached_property
    def sweepfile(self) -> SweepFile:
        spec = importlib.util.spec_from_file_location(
            self.full_name, str(self.path / f"{self.module_name}.py")
        )
        sweepfile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweepfile)
        return sweepfile

    def initialize_sweep(self):
        dump = self.sweepfile.cfg.sweep()
        str(dump)
        sweep_id = wandb.sweep(
            sweep={
                "parameters": dump,
                "method": "grid",
            },
            project=self.sweepfile.PROJECT,
        )
        f = open(self.path / "sweep_id.txt", "w")
        f.write(sweep_id)
        f.close()

    @property
    def sweep_id(self):
        return open(self.path / "sweep_id.txt").read().strip()

    def run(self):
        wandb.init()
        basecfg: SweepableConfig = self.sweepfile.cfg

        cfg = basecfg.from_selective_sweep(dict(wandb.config))
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
        self.sweepfile.run(cfg)
        wandb.finish()

    def start_agent(self):
        wandb.agent(
            self.sweep_id,
            function=self.run,
            project=self.sweepfile.PROJECT,  # TODO change project to being from config maybe? or remove from config
        )

    def rand_run_no_agent(self, init=False):
        basecfg: SweepableConfig = self.sweepfile.cfg

        cfg = basecfg.random_sweep_configuration()
        # wandb.init()
        if init:
            wandb.init(config=cfg.model_dump(), project=self.sweepfile.PROJECT)
        # wandb.config.update(cfg.model_dump())
        self.sweepfile.run(cfg)
        wandb.finish()


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
