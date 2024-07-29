import wandb
from wandb.apis.public import Api, Runs, Run, Sweep, Project
from pathlib import Path
import pandas as pd

api = wandb.Api()


class RunHistories:
    def __init__(self, path="~/workspace/wandb_run_history_caches"):
        self.path = Path(path)
        self.runs = {}
    def get_run_dir(self, run):
        return self.path / run.project / run.sweep.id / "history" / run.id 
    def get_run_key_path(self, run, key):
        return (
           / f"{key}.csv"
        )

    def download(self, run, key):
        hist = run.scan_history(keys=[key, "_step"])
        df = pd.DataFrame([i for i in hist])
        path = self.get_run_key_path(run, key)
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / f"{key}.csv")

    def load(self, run, key):
        if run not in self.runs or key in self.runs[run]:
            self.runs[run][key] = pd.read_csv(self.get_run_key_path(run, key))
        return self.runs[run][key]

    def get_runs(self, runs: Runs, keys):
        donthave = {}
        for key in keys:
            donthave[key] = []
            for run in runs:
                id = (run.project, run.sweep.id, run.id)
                if self.get_run_key_path(run, key).exists():
                    self.runs[id] = self.get_run_key_path(run, key)

                #TODO Pick up here
                
                # cache_path = self.path / run.project / run.sweep.id / run.id
                # cache_path.mkdir(parents=True, exist_ok=True)
                run: Run
                print(run.name)
                print(run.id)
                print(run.storage_id)


sweep = api.sweep("sae sweeps/5uwxiq76")
runs = sweep.runs

rh = RunHistories()
rh.get_runs(runs)
