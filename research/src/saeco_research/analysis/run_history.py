import subprocess
import time
from pathlib import Path

import pandas as pd
import wandb
from wandb.apis.public import Run, Runs

api = wandb.Api()

home = Path.home()


class RunHistories:
    def __init__(self, path=home / "workspace/wandb_run_history_caches"):
        self.path = Path(path)
        self.runs = {}

    def get_run_dir(self, run):
        return (
            self.path
            / run.project.replace(" ", "-")
            / run.sweep.id
            / "history"
            / run.id
        )

    def get_run_key_path(self, run, key):
        key = key.replace("/", ".")
        return self.get_run_dir(run) / f"{key}.csv"

    def download(self, run, key):
        hist = run.scan_history(keys=[key, "_step"])
        df = pd.DataFrame([i for i in hist])
        self.get_run_dir(run).mkdir(parents=True, exist_ok=True)
        df.to_csv(self.get_run_key_path(run, key))
        print("Downloaded", run.name, key)

    def load(self, run, key):
        runid = (run.project, run.sweep.id, run.id)
        if runid not in self.runs:
            raise ValueError("Run was not loaded")
        if key not in self.runs[runid]:
            self.runs[runid][key] = pd.read_csv(self.get_run_key_path(run, key))
        return self.runs[runid][key]

    def get(self, run, key):
        id = (run.project, run.sweep.id, run.id)

    def get_runs(self, runs: Runs, keys):
        donthave = {}
        procs = []
        for key in keys:
            donthave[key] = []
            for run in runs:
                id = (run.project, run.sweep.id, run.id)
                if not self.get_run_key_path(run, key).exists():
                    print("Starting downloading", run.name, key)
                    procs.append(self.dispatch(run, key))
                    print(run.name)
                    print(run.id)
                    print(run.storage_id)
                self.runs[id] = {}

                # TODO Pick up here

                # cache_path = self.path / run.project / run.sweep.id / run.id
                # cache_path.mkdir(parents=True, exist_ok=True)
                run: Run
        ppoll = [True]
        key
        len([self.get_run_dir(run).exists() for run in runs])
        len(
            list(
                (
                    self.path / run.project.replace(" ", "-") / run.sweep.id / "history"
                ).iterdir()
            )
        )

        d = {run.id: [] for run in runs}
        for run in runs:
            d[run.id].append(run)
        r1, r2 = d["4ki37bgq"]
        r1.path
        r2.path
        r1: Run
        r1.name
        r2.name
        # d["3frlnlq9"]
        while any(ppoll):
            time.sleep(1)
            ppoll = [p.poll() is None for p in procs]
            print(f"waiting on {sum(ppoll)} procs")
        print("Done")
        for p in procs:
            print(p.communicate())

    def get_runs2(self, runs: Runs, keys):
        donthave = {}
        procs = []
        for key in keys:
            donthave[key] = []
            for run in runs:
                id = (run.project, run.sweep.id, run.id)
                if not self.get_run_key_path(run, key).exists():
                    print("Starting downloading", run.name, key)
                    procs.append(self.dispatch(run, key))
                self.runs[id] = {}

                # TODO Pick up here

                # cache_path = self.path / run.project / run.sweep.id / run.id
                # cache_path.mkdir(parents=True, exist_ok=True)
                run: Run
                print(run.name)
                print(run.id)
                print(run.storage_id)
        ppoll = [True]
        while any(ppoll):
            time.sleep(1)
            ppoll = [p.poll() is None for p in procs]
            print(f"waiting on {sum(ppoll)} procs")
        print("Done")
        for p in procs:
            print(p.communicate())

    def dispatch(self, run, key) -> subprocess.CompletedProcess:
        run_str = "/".join(run.path).replace("+", " ")
        cmd = " ".join(
            [
                "/bin/python3",
                __file__,
                "--run-path",
                f"'{run_str}'",
                "--key",
                f"'{key}'",
            ]
        )
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        return p


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", type=str)
    parser.add_argument("--key", type=str)

    args = parser.parse_args()

    if args.run_path:
        assert args.key, "Must provide a key to download"
        print(args.key)
        print(args.run_path)
        # exit()
        run = api.run(args.run_path)
        rh = RunHistories()
        rh.download(run, args.key)
    # else:

    #     sweep = api.sweep("sae sweeps/5uwxiq76")
    #     runs = sweep.runs
    #     print(runs[0].path)
    #     api.run("/".join(runs[0].path).replace("+", " "))
    #     rh = RunHistories()
    #     rh.get_runs(runs[0:8], ["cache/L2_loss"])
    #     print(__file__)


if __name__ == "__main__":
    main()
