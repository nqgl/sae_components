from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.tosweepfields import (
    ParametersToSweep,
    SweepableNode,
)
from saeco.trainer.run_config import RunConfig

# from .model import Config


from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig


class SubConfig(SweepableConfig):
    a: int


class Config(SweepableConfig):
    a: int
    b: int
    a_b: int
    c: SubConfig


PROJECT = "sae sweeps"

var = SweepVar(1, 2, 3, name="var")


cfg = Config(
    a=SweepExpression(var, expr=lambda x: x),
    b=Swept(1, 2, 3),
    a_b=SweepExpression(var, expr=lambda x: x + 2),
    c=SubConfig(a=Swept(1, 2, 3)),
)

ps = ParametersToSweep.from_config(cfg)
print(ps)
print()
