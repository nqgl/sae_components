import importlib
from pathlib import Path
import wandb
from saeco.misc import lazyprop


class Sweeper:
    def __init__(self, path):
        self.path = Path(path)

    @lazyprop
    def sweepfile(self):
        spec = importlib.util.spec_from_file_location(
            "sweepfile", str(self.path / "sweepfile.py")
        )
        sweepfile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweepfile)
        return sweepfile

    def initialize_sweep(self):
        dump = self.sweepfile.cfg.sweep()
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
        basecfg = self.sweepfile.cfg

        cfg = basecfg.from_selective_sweep(dict(wandb.config))
        wandb.config.update(cfg.model_dump())
        print(dict(wandb.config))
        self.sweepfile.run(cfg)
        wandb.finish()

    def start_agent(self):
        wandb.agent(
            self.sweep_id,
            function=self.run,
            project=self.sweepfile.PROJECT,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweeper for Saeco")
    parser.add_argument("path", type=str)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--runpod-n-instances", type=int)
    args = parser.parse_args()
    sw = Sweeper(args.path)
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
