from pydantic import Field

from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer.normalizers import GNConfig
from saeco.trainer.train_config import TrainConfig


class RunConfig[ArchCfgT: SweepableConfig](SweepableConfig):
    train_cfg: TrainConfig
    arch_cfg: ArchCfgT
    normalizer_cfg: GNConfig = Field(default_factory=GNConfig)
    resampler_config: AnthResamplerConfig = Field(default_factory=AnthResamplerConfig)
    init_cfg: InitConfig = Field(default_factory=InitConfig)
