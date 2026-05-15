from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.sweep_data import SweepData
from attrs import define, field
from sweepable import SweepableConfig


@define
class RunConfig:
    config: dict = field(factory=dict)


class CustomSweeper:
    root_config: SweepableConfig

    def __init__(self, prev_logger):
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
            self.root_config.sweep_info_tree.select_instance_by_index(sweep_index)
        )
        cfg: SweepableConfig = self.root_config.from_selective_sweep(
            selective_instance_sweep_dict
        )
        if cfg.get_hash() != sweep_hash:
            wow_match_weird = None
            for i in range(
                self.root_config.sweep_info_tree.swept_combinations_count_including_vars()
            ):
                h = self.root_config.from_selective_sweep(
                    self.root_config.sweep_info_tree.select_instance_by_index(i)
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


class WandbCustomLogger:
    root_config: "SweepableConfig"

    def __init__(self):
        from wandb.sdk.wandb_run import Run

        import wandb

        self.wandb = wandb
        self.run: Run | None = None
        self.project = "default-project"
        self.run_name: str | None = None

        self.root_config: SweepableConfig = SweepableConfig()
        self._sweep_inst_config: dict | None = None
        self.run_config = RunConfig()

        # cache of namespaced dicts, to emulate neptune exists/fetch/assign
        self._namespaces: dict[str, dict] = {}

    # ---------- lifecycle ----------

    def init(
        self,
        project: str | None = None,
        config: dict | None = None,
        run_name: str | None = None,
    ):
        assert self.run is None, "Run already initialized"
        if project:
            self.project = project
        self.run_name = run_name

        print("init wandb", self.project, run_name)
        self.run = self.wandb.init(project=self.project, name=run_name)
        print("init wandb done")

        if config:
            self.update_config(config)

    def finish(self):
        assert self.run is not None, "No active run"
        self.wandb.finish()
        self.run = None
        self.run_name = None
        self._namespaces.clear()

    def log(self, data: dict, step: int | None = None):
        """
        Preserve Neptune's behavior:
          - Rename leaf key `foo/bar` -> `foo/bar_`
          - Append-like semantics via wandb.log
        """
        assert self.run is not None, "Call init() first"

        transformed: dict[str, object] = {}
        for key, value in data.items():
            ks = key.split("/")
            new_key = "/".join(ks[:-1] + [f"{ks[-1]}_"])
            transformed[new_key] = value
        if step is not None:
            self.wandb.log(transformed, step=step)
        else:
            self.wandb.log(transformed)

    # @classmethod
    # def _wandb_config_fix(cls, item):
    #     from enum import Enum

    #     if isinstance(item, Enum):
    #         return item.value
    #     if isinstance(item, dict):
    #         return {k: cls._wandb_config_fix(v) for k, v in item.items()}
    #     if isinstance(item, (list, tuple)):
    #         return {i: cls._wandb_config_fix(v) for i, v in enumerate(item)}
    #     return item

    @classmethod
    def wandb_config_fix(cls, item):
        return item
        # try:
        #     import wandb
        #     fixed = cls._wandb_config_fix(item)
        #     return wandb.util.json_friendly(fixed)
        # except Exception:
        #     return str(item)

    def update_config(self, config_dict: dict):
        self.update_namespace("config", config_dict)

    def update_namespace(self, namespace: str, data: dict):
        if self.run is None:
            raise ValueError("update_namespace called before init")

        existing = self._namespaces.get(namespace)
        if existing is None:
            try:
                existing = dict(self.run.summary.get(namespace, {}))
            except Exception:
                existing = {}

        merged = {**existing, **data}
        fixed = self.wandb_config_fix(merged)
        self.run.config[namespace] = fixed
        # self.run.summary[namespace] = fixed
        self._namespaces[namespace] = (
            dict(fixed) if isinstance(fixed, dict) else {"value": fixed}
        )

        flat_update = {f"{namespace}/{k}": v for k, v in merged.items()}
        self.run_config.config.update(flat_update)

        try:
            self.run.config.update(
                self.wandb_config_fix(flat_update), allow_val_change=True
            )
        except Exception:
            for k, v in flat_update.items():
                try:
                    self.run.config.update(
                        {k: self.wandb_config_fix(v)}, allow_val_change=True
                    )
                except Exception:
                    pass

    # ---------- sweep helpers ----------

    def agent(
        self,
        sweep_data: "SweepData",
        function,
        sweep_index: int,
        sweep_hash: str,
        project: str | None = None,
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
        sweep_index: int,
        sweep_hash: str,
        project: str | None = None,
    ):
        self.root_config = sweep_data.root_arch_ref.config
        selective_instance_sweep_dict = (
            self.root_config.sweep_info_tree.select_instance_by_index(sweep_index)
        )
        cfg = self.root_config.from_selective_sweep(selective_instance_sweep_dict)

        if cfg.get_hash() != sweep_hash:
            wow_match_weird = None
            for i in range(
                self.root_config.sweep_info_tree.swept_combinations_count_including_vars()
            ):
                h = self.root_config.from_selective_sweep(
                    self.root_config.sweep_info_tree.select_instance_by_index(i)
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
        sweep_index: int,
        sweep_hash: str,
    ):
        assert self.run is not None, "Call init() first"

        selective_instance_sweep_dict = selective_instance_sweep_dict.copy()
        payload = {
            "swept": selective_instance_sweep_dict,
            "sweep_id": sweep_data.sweep_id,
            "sweep_index": sweep_index,
            "sweep_hash": sweep_hash,
            "sweep_expressions": self.root_config.get_sweepexpression_instantiations(
                selective_instance_sweep_dict
            ),
        }
        self.update_namespace("sweep", payload)

    def config_get(self) -> dict | None:
        return self._sweep_inst_config


from attrs import define, field


@define
class BaseLogger[RunT]:
    root_config: SweepableConfig
    run: RunT | None = None
    run_name: str | None = None
    _sweep_inst_config = None
    DEFAULT_PROJECT_STR: str = "default-project"

    def init(
        self,
        project: str | None = None,
        config: dict | None = None,
        run_name: str | None = None,
    ):
        assert self.run is None
        self.run_name = run_name
        if project:
            self.project = project
        self.run = self._create_run(project, run_name)
        if config:
            self.update_config(config)

    def _create_run(self, project: str | None, run_name: str | None) -> RunT:
        raise NotImplementedError("Subclasses must implement _create_run")

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

    def update_config(self, config_dict: dict):
        raise NotImplementedError("Subclasses must implement update_config")

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
            self.root_config.sweep_info_tree.select_instance_by_index(sweep_index)
        )
        cfg: SweepableConfig = self.root_config.from_selective_sweep(
            selective_instance_sweep_dict
        )
        if cfg.get_hash() != sweep_hash:
            wow_match_weird = None
            for i in range(
                self.root_config.sweep_info_tree.swept_combinations_count_including_vars()
            ):
                h = self.root_config.from_selective_sweep(
                    self.root_config.sweep_info_tree.select_instance_by_index(i)
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

        self.update_run_with_sweep_data(
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

    def update_run_with_sweep_data(self, data: dict):
        raise NotImplementedError(
            "Subclasses must implement update_run_with_sweep_data"
        )

    def config_get(self): ...
