from functools import cached_property

from pydantic import Field

from saeco.components.sae_cache import SAECache

from saeco.data.data_cfg import DataConfig
from saeco.misc import lazycall
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from .OptimConfig import get_optim_cls
from .schedule_cfg import RunSchedulingConfig


class EarlyStoppingBounds(SweepableConfig):
    # check_timestamps: list[int] = Field(default_factory=list[int])
    min_values: dict[str, dict[int, float]]
    max_values: dict[str, dict[int, float]]

    @classmethod
    def none(cls):
        return cls(min_values={}, max_values={})

    @cached_property
    def check_timestamps(self):
        return list(
            set.union(
                *[
                    set(d.keys())
                    for d in list(self.min_values.values())
                    + list(self.max_values.values())
                ]
            )
        )

    def should_stop(self, cache: SAECache, t: int):
        if t not in self.check_timestamps:
            return False
        print("getfields", cache._getfields())
        for k, v in self.min_values.items():
            if t not in v:
                continue
            v = v[t]
            if cache.get(k) < v:
                return True
        for k, v in self.max_values.items():
            if t not in v:
                continue
            v = v[t]
            if cache.get(k) > v:
                return True
        return False


class TrainConfig(SweepableConfig):
    data_cfg: DataConfig = Field(default_factory=DataConfig)
    wandb_cfg: dict = Field(default_factory=lambda: dict(project="sae sweeps"))
    coeffs: dict[str, float] = Field(default_factory=lambda: dict(sparsity_loss=1e-3))
    # coeffs: Coeffs = Field(default_factory=Coeffs)
    l0_targeter_type: str = "gentle_basic"
    l0_target: float | None = None
    l0_targeting_enabled: bool = True
    l0_target_adjustment_size: float = 0.0003
    use_autocast: bool = True
    batch_size: int = 4096
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    use_lars: bool = False
    kwargs: dict = Field(default_factory=dict)
    optim: str = "RAdam"
    raw_schedule_cfg: RunSchedulingConfig = Field(default_factory=RunSchedulingConfig)
    use_averaged_model: bool = True
    checkpoint_period: int | None = None
    save_on_complete: bool = True
    weight_decay: float | None = None
    intermittent_metric_freq: int = 1000

    input_sites: list[str] | None = None
    target_sites: list[str] | None = None
    early_stopping_bounds: EarlyStoppingBounds = Field(
        default_factory=EarlyStoppingBounds.none
    )

    @property
    @lazycall
    def schedule(self):
        return self.raw_schedule_cfg.step_scheduler

    @property
    def use_schedulefree(self):
        return self.optim == "ScheduleFree"

    def get_optim(self):
        return get_optim_cls(self.optim)

    def get_databuffer(self, num_workers=None):
        if num_workers is None:
            num_workers = self.data_cfg.databuffer_num_workers
        if self.input_sites and not (  # <= checks subset
            set(self.input_sites) <= set(self.data_cfg.model_cfg.acts_cfg.sites)
        ):
            raise ValueError(
                f"Input sites must be a subset of the data config's sites. Got {self.input_sites}, expected subset of {self.data_cfg.model_cfg.acts_cfg.sites}"
            )

        if self.target_sites and not (
            set(self.target_sites) <= set(self.data_cfg.model_cfg.acts_cfg.sites)
        ):
            raise ValueError(
                f"Target sites must be a subset of the data config's sites. Got {self.target_sites}, expected subset of {self.data_cfg.model_cfg.acts_cfg.sites}"
            )

        used_input_sites = self.input_sites or self.data_cfg.model_cfg.acts_cfg.sites

        buffer = self.data_cfg._get_queued_databuffer(
            num_workers=num_workers,
            batch_size=self.batch_size,
            input_sites=used_input_sites,
            target_sites=self.target_sites,
        )

        return buffer
