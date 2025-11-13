from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.trainer.run_config import RunConfig
from .model import Config
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "sae sweeps"

var = SweepVar(1, 2, 3, name="var")


cfg = Config(
    a=SweepExpression(var, expr=lambda x: x),
    b=SweepExpression(var, expr=lambda x: x + 1),
    a_b=SweepExpression(var, expr=lambda x: x + 2),
)
