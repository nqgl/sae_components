import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig

from saeco.sweeps.sweepable_config.SweptNode import SweptNode

# Environment variables for logger selection
WANDB = os.environ.get("USE_WANDB", "false").lower() in ("true", "1", "yes")
USE_NEPTUNE = os.environ.get("USE_NEPTUNE", "true").lower() in ("true", "1", "yes")
CUSTOM_SWEEP = os.environ.get("CUSTOM_SWEEP", "false").lower() in ("true", "1", "yes")


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

    def agent(self, sweep_data: "SweepData", function, project=None):
        sweep_id = sweep_data.sweep_id
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


class NeptuneLogger(Logger):
    def __init__(self):
        import neptune

        self.neptune = neptune
        self.run: neptune.Run | None = None
        self.project = "default-project"

    @classmethod
    def neptune_config_fix(cls, item):
        from enum import Enum

        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict):
            return {k: cls.neptune_config_fix(v) for k, v in item.items()}
        elif isinstance(item, list) or isinstance(item, tuple):
            return {i: v for i, v in enumerate(item)}

        return item

    def init(self, project=None, config=None, run_name=None):
        import neptune

        assert self.run is None
        if project:
            self.project = project
        print("init neptune", self.project, run_name)
        self.run = neptune.init_run(
            project=self.project,
            name=run_name,
        )
        print("init neptune done")
        if config:
            self.update_config(config)

    def log(self, data, step=None):
        for key, value in data.items():
            key = "/".join((ks := key.split("/"))[:-1] + [f"{ks[-1]}_"])
            if step is not None:
                self.run[key].append(value, step=step)
            else:
                self.run[key].append(value)

    def finish(self):
        assert self.run is not None
        self.run.stop()
        self.run = None

    def update_config(self, config_dict):
        self.update_namespace("config", config_dict)

    def update_namespace(self, namespace, data: dict):
        if self.run.exists(namespace):
            data = {**self.run[namespace].fetch(), **data}
        self.run[namespace].assign(self.neptune_config_fix(data))

    def sweep(self, swept_nodes: SweptNode, project):
        raise NotImplementedError("Neptune sweeps not yet implemented")

    def agent(self, sweep_data: "SweepData", function, project=None):
        raise NotImplementedError("Neptune sweeps not yet implemented")

    def config_get(self):
        print("config_get neptune")
        if self.run and "parameters" in self.run:
            print("config_get neptune", self.run["parameters"].fetch())
            return dict(self.run["parameters"].fetch())
        print("config_get neptune empty")
        return {}


if TYPE_CHECKING:
    from saeco.architecture.arch_reload_info import ArchRef


class CustomSweeper(Logger):
    root_config: SweepableConfig

    def __init__(self, prev_logger: NeptuneLogger):
        # self._config: SweepableConfig = None
        self.root_config: SweepableConfig = SweepableConfig()
        self._sweep_inst_config = None
        # self.sweep_data: "SweepData" | None = None
        self.prev_logger = prev_logger
        assert not isinstance(prev_logger, CustomSweeper)

    def init(self, project=None, config=None, run_name=None):
        assert config is None
        self.prev_logger.init(project=project, config=config, run_name=run_name)

    def log(self, data, step=None):
        self.prev_logger.log(data, step=step)

    def finish(self):
        self.prev_logger.finish()

    def update_config(self, config_dict):
        self.prev_logger.update_config(config_dict)

    def sweep(self, sweep_config, project): ...

    def agent(
        self,
        sweep_data: "SweepData",
        function,
        sweep_index,
        sweep_hash,
        project=None,
    ):
        self.root_config: SweepableConfig = sweep_data.root_arch_ref.config
        selective_instance_sweep_dict = (
            self.root_config.to_swept_nodes().select_instance_by_index(sweep_index)
        )
        cfg: SweepableConfig = self.root_config.from_selective_sweep(
            selective_instance_sweep_dict
        )
        assert cfg.get_hash() == sweep_hash, f"{cfg.get_hash()} != {sweep_hash}"
        self._sweep_inst_config = selective_instance_sweep_dict
        if project:
            self.prev_logger.project = project

        function()

    def log_sweep(
        self,
        selective_instance_sweep_dict: dict,
        sweep_data: "SweepData",
        sweep_index,
        sweep_hash,
    ):
        selective_instance_sweep_dict = selective_instance_sweep_dict.copy()

        self.prev_logger.update_namespace(
            "sweep",
            {
                "swept": selective_instance_sweep_dict,
                "sweep_id": sweep_data.sweep_id,
                "sweep_index": sweep_index,
                "sweep_hash": sweep_hash,
                "sweep_expressions": self.root_config.get_sweepexpression_instantiations(
                    selective_instance_sweep_dict
                ),
            },
        )

    def config_get(self):
        return self._sweep_inst_config


def get_logger():
    if USE_NEPTUNE:
        return NeptuneLogger()
    elif WANDB:
        return WandbLogger()
    else:
        return NoOpLogger()


# logger_instance = get_logger()
# if CUSTOM_SWEEP:
#     logger_instance = CustomSweeper(logger_instance)


# @contextmanager
# def enter(project=None, config=None, run_name=None):
#     logger_instance.init(project=project, config=config, run_name=run_name)
#     yield
#     logger_instance.finish()
