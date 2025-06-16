import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData
from attrs import define, field

from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig

from saeco.sweeps.sweepable_config.SweptNode import SweptNode
from .neptune_scale_metric_logger import NeptuneScaleMetricLogger


@define
class RunConfig:
    config: dict = field(factory=dict)


class NeptuneLogger:
    def __init__(self):
        import neptune

        self.neptune = neptune
        self.run: neptune.Run | None = None
        self.project = "default-project"
        self.run_name = None
        self.run_config = RunConfig()

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
        self.run_name = run_name
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
        self.run_name = None

    def update_config(self, config_dict):
        self.update_namespace("config", config_dict)

    def update_namespace(self, namespace, data: dict):
        if self.run.exists(namespace):
            data = {**self.run[namespace].fetch(), **data}
        self.run[namespace].assign(self.neptune_config_fix(data))
        self.run_config.config.update({f"{namespace}/{k}": v for k, v in data.items()})

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


class CustomSweeper:
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
        if cfg.get_hash() != sweep_hash:
            wow_match_weird = None
            for i in range(
                self.root_config.to_swept_nodes().swept_combinations_count_including_vars()
            ):
                h = self.root_config.from_selective_sweep(
                    self.root_config.to_swept_nodes().select_instance_by_index(i)
                ).get_hash()
                if h == sweep_hash:
                    wow_match_weird = i
                    break
            if wow_match_weird is not None:
                raise ValueError(
                    f"{cfg.get_hash()} != {sweep_hash} but {wow_match_weird} matches"
                )
            raise ValueError(
                f"{cfg.get_hash()} != {sweep_hash}, and didn't match other options either"
            )
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


class NeptuneCustomLogger:
    root_config: SweepableConfig

    def __init__(self):
        # self._config: SweepableConfig = None
        self.root_config: SweepableConfig = SweepableConfig()
        self._sweep_inst_config = None
        # self.sweep_data: "SweepData" | None = None
        import neptune

        self.neptune = neptune
        self.run: neptune.Run | None = None
        self.project = "default-project"
        self.run_name = None
        self.run_config = RunConfig()

    def init(self, project=None, config=None, run_name=None):
        import neptune

        assert self.run is None
        self.run_name = run_name
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
        self.run_name = None

    @classmethod
    def _neptune_config_fix(cls, item):
        from enum import Enum

        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict):
            return {k: cls.neptune_config_fix(v) for k, v in item.items()}
        elif isinstance(item, list) or isinstance(item, tuple):
            return {i: v for i, v in enumerate(item)}
        return item

    @classmethod
    def neptune_config_fix(cls, item):
        from .neptune_scale_metric_logger import stringify_unsupported

        return stringify_unsupported(cls._neptune_config_fix(item))

    def update_config(self, config_dict):
        self.update_namespace("config", config_dict)

    def update_namespace(self, namespace, data: dict):
        if self.run is not None and self.run.exists(namespace):
            data = {**self.run[namespace].fetch(), **data}
        self.run[namespace].assign(self.neptune_config_fix(data))
        self.run_config.config.update({f"{namespace}/{k}": v for k, v in data.items()})

    def agent(
        self,
        sweep_data: "SweepData",
        function,
        sweep_index,
        sweep_hash,
        project=None,
    ):
        self.prepare_sweep(
            sweep_data=sweep_data,
            sweep_index=sweep_index,
            sweep_hash=sweep_hash,
            project=project,
        )

        function()

    def prepare_sweep(
        self,
        sweep_data: "SweepData",
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
        if cfg.get_hash() != sweep_hash:
            wow_match_weird = None
            for i in range(
                self.root_config.to_swept_nodes().swept_combinations_count_including_vars()
            ):
                h = self.root_config.from_selective_sweep(
                    self.root_config.to_swept_nodes().select_instance_by_index(i)
                ).get_hash()
                if h == sweep_hash:
                    wow_match_weird = i
                    break
            if wow_match_weird is not None:
                raise ValueError(
                    f"{cfg.get_hash()} != {sweep_hash} but {wow_match_weird} matches"
                )
            raise ValueError(
                f"{cfg.get_hash()} != {sweep_hash}, and didn't match other options either"
            )
        self._sweep_inst_config = selective_instance_sweep_dict
        if project:
            self.project = project

    def log_sweep(
        self,
        selective_instance_sweep_dict: dict,
        sweep_data: "SweepData",
        sweep_index,
        sweep_hash,
    ):
        selective_instance_sweep_dict = selective_instance_sweep_dict.copy()

        self.update_namespace(
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


import neptune_scale


class NeptuneScaleLogger(NeptuneCustomLogger):
    run: neptune_scale.Run | None

    @property
    def runlogger(self):
        if self.run is None:
            return None
        return NeptuneScaleMetricLogger(self.run, "")

    def init(self, project=None, config=None, run_name=None):
        import neptune_scale

        assert self.run is None
        self.run_name = run_name
        if project:
            self.project = project

        self.run = neptune_scale.Run(
            project=self.project,
            run_id=run_name,
            experiment_name=run_name.split(":")[0] if ":" in run_name else run_name,
            # experiment_name=run_name,
        )

        if config:
            self.update_config(config)

    def log(self, data, step=None):
        for key, value in data.items():
            key = "/".join((ks := key.split("/"))[:-1] + [f"{ks[-1]}_"])
            if step is not None:
                self.runlogger[key].log_metric(value, step=step)
            else:
                self.runlogger[key] = value

    def finish(self):
        assert self.run is not None
        self.run.close()
        self.run = None
        self.run_name = None

    def update_namespace(self, namespace, data: dict):
        self.runlogger[namespace] = data
