from typing import Generic, TypeVar

from pydantic import Field

from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer.normalizers import GNConfig
from saeco.trainer.train_config import TrainConfig

T = TypeVar("T", bound=SweepableConfig)


class RunConfig(SweepableConfig, Generic[T]):
    train_cfg: TrainConfig
    arch_cfg: T
    normalizer_cfg: GNConfig = Field(default_factory=GNConfig)
    resampler_config: AnthResamplerConfig = Field(default_factory=AnthResamplerConfig)
    init_cfg: InitConfig = Field(default_factory=InitConfig)
