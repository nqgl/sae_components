from saeco.data.dataset import DataConfig
from saeco.misc import lazycall
from saeco.sweeps import SweepableConfig, Swept
from saeco.trainer.OptimConfig import get_optim_cls
from saeco.trainer.schedule_cfg import RunSchedulingConfig


from pydantic import Field


class TrainConfig(SweepableConfig):
    data_cfg: DataConfig = Field(default_factory=DataConfig)
    wandb_cfg: dict = Field(default_factory=lambda: dict(project="sae sweeps"))
    coeffs: dict[str, float | Swept[float]] = Field(
        default_factory=lambda: dict(sparsity_loss=1e-3)
    )
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

    @property
    @lazycall
    def schedule(self):
        return self.raw_schedule_cfg.step_scheduler

    @property
    def use_schedulefree(self):
        return self.optim == "ScheduleFree"

    def get_optim(self):
        return get_optim_cls(self.optim)
