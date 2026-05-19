from pydantic import Field

from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer.normalizers import GNConfig
from saeco.trainer.train_config import TrainConfig


class RunConfigBase[ArchCfgT: SweepableConfig](SweepableConfig):
    arch_cfg: ArchCfgT


class RunConfig[ArchCfgT: SweepableConfig](RunConfigBase[ArchCfgT]):
    """The complete specification for a run (or a sweep of runs).

    Bundles the four sub-configs an ``Architecture`` needs:
    ``arch_cfg`` (architecture-specific), ``train_cfg`` (data, schedule,
    optimizer, loss coefficients), ``init_cfg`` (dictionary sizing),
    ``resampler_config`` and ``normalizer_cfg``. Being a
    ``SweepableConfig`` itself, any field at any depth may be a sweep.
    """

    train_cfg: TrainConfig
    normalizer_cfg: GNConfig = Field(default_factory=GNConfig)
    resampler_config: AnthResamplerConfig = Field(default_factory=AnthResamplerConfig)
    init_cfg: InitConfig = Field(default_factory=InitConfig)
