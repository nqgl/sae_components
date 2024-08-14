from saeco.components.resampling.anthropic_resampling import AnthResamplerConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer.normalizers import GNConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.trainer.TrainConfig import TrainConfig


from pydantic import Field


from typing import Generic, TypeVar

T = TypeVar("T", bound=SweepableConfig)


class RunConfig(SweepableConfig, Generic[T]):
    train_cfg: TrainConfig
    arch_cfg: T
    normalizer_cfg: GNConfig = Field(default_factory=GNConfig)
    resampler_config: AnthResamplerConfig = Field(default_factory=AnthResamplerConfig)
    sae_cfg: InitConfig = Field(default_factory=InitConfig)
