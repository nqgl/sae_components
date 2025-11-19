from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression

from .model import Config

PROJECT = "sae sweeps"

var = SweepVar(1, 2, 3, name="var")


cfg = Config(
    a=SweepExpression(var, expr=lambda x: x),
    b=SweepExpression(var, expr=lambda x: x + 1),
    a_b=SweepExpression(var, expr=lambda x: x + 2),
)
