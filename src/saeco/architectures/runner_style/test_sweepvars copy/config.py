from saeco.sweeps.sweepable_config.tosweepfields import (
    ParametersToSweep,
)

# from .model import Config
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept


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
