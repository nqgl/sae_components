import os
from functools import cached_property
from pathlib import Path
from typing import Protocol
from saeco.misc import lazyprop
from saeco.sweeps.sweepable_config import SweepableConfig
from attrs import define, field
from saeco.mlog import mlog


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
        self.pods = None

    @cached_property
    def sweepfile(self) -> SweepFile:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            self.full_name, str(self.path / f"{self.module_name}.py")
        )
        sweepfile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweepfile)
        return sweepfile

    def initialize_sweep(self):
        cfg = self.sweepfile.cfg
        representation = cfg.to_swept_nodes()
        sweep_id = mlog.begin_sweep(
            representation.to_wandb(), project=self.sweepfile.PROJECT
        )
        with open(self.path / "sweep_id.txt", "w") as f:
            f.write(sweep_id)

    @property
    def sweep_id(self):
        return open(self.path / "sweep_id.txt").read().strip()

    def run(self):
        basecfg: SweepableConfig = self.sweepfile.cfg
        current_config = mlog.config()
        cfg = basecfg.from_selective_sweep(dict(current_config))

        # Instead of wandb.config.update(...), now use mlog.update_config
        # to store run metadata if needed
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
        mlog.update_config(dict(full_cfg=cfg.model_dump(), pod_info=pod_info))

        self.sweepfile.run(cfg)
        mlog.finish()

    def start_agent(self):
        mlog.run_agent(self.sweep_id, self.sweepfile.PROJECT, self.run)

    def rand_run_no_agent(self):
        basecfg: SweepableConfig = self.sweepfile.cfg
        cfg = basecfg.random_sweep_configuration()
        # For a random run, just init mlog and run
        with mlog.enter(project=self.sweepfile.PROJECT, config=cfg.model_dump()):
            self.sweepfile.run(cfg)

    def start_pods(self, num_pods=None):
        from ezpod import Pods

        pods = Pods.All()
        if num_pods:
            pods.make_new_pods(num_pods)
        pods.sync()
        pods.setup()
        pods.runpy(f"src/saeco/sweeps/sweeper2.py --worker --sweep-id {self.sweep_id}")
        pods.purge()


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
