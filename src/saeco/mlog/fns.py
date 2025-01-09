import os
from contextlib import contextmanager

from saeco.sweeps.sweepable_config.SweptNode import SweptNode

# Instead of hardcoding WANDB = True, let's read from config or env
WANDB = os.environ.get("USE_WANDB", "true").lower() in ("true", "1", "yes")


class Logger:
    """Generic logger interface."""

    def init(self, project=None, config=None, run_name=None): ...
    def log(self, data, step=None): ...
    def finish(self): ...
    def update_config(self, config_dict): ...
    def sweep(self, sweep_config, project): ...
    def agent(self, sweep_id, project, function): ...
    def config_get(self): ...


class WandbLogger(Logger):
    def __init__(self):
        import wandb

        self.wandb = wandb
        self.run = None
        self.project = "default_project"

    def init(self, project=None, config=None, run_name=None):
        self.run = self.wandb.init(
            project=project or self.project, config=config, name=run_name
        )

    def log(self, data, step=None):
        if step is not None:
            self.wandb.log(data, step=step)
        else:
            self.wandb.log(data)

    def finish(self):
        if self.wandb.run:
            self.wandb.finish()

    def update_config(self, config_dict):
        self.wandb.config.update(config_dict)

    def sweep(self, swept_nodes: SweptNode, project):
        res = self.wandb.sweep(
            sweep=swept_nodes.to_wandb(), project=project or self.project
        )
        return res

    def agent(self, sweep_id, function, project=None):
        print("STARTING AGENT:", sweep_id, project, self.project)
        self.wandb.agent(sweep_id, function=function, project=project or self.project)

    def config_get(self):
        # wandb.config is a wandb process global
        return dict(self.wandb.config)


class NoOpLogger(Logger):
    def __init__(self):
        self._config = {}

    def init(self, project=None, config=None, run_name=None):
        if config is not None:
            self._config.update(config)

    def log(self, data, step=None):
        pass  # no-op

    def finish(self):
        pass

    def update_config(self, config_dict):
        self._config.update(config_dict)

    def sweep(self, sweep_config, project):
        # No real sweep, just return a fake sweep_id
        return "no_sweep_id"

    def agent(self, sweep_id, project, function):
        # Just run function once with current config
        function()

    def config_get(self):
        return self._config


def get_logger():
    if WANDB:
        return WandbLogger()
    else:
        return NoOpLogger()


logger_instance = get_logger()


@contextmanager
def enter(project=None, config=None, run_name=None):
    logger_instance.init(project=project, config=config, run_name=run_name)
    yield
    logger_instance.finish()
