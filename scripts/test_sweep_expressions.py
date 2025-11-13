from saeco.sweeps.sweepable_config.sweep_expressions import (
    ExpressionOpEnum,
    Op,
    Val,
    SweepVar,
    Swept,
)
from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config import SweepableConfig


class Cfg(SweepableConfig):
    a: int
    b: float
    c: float
    d: float


device_microbatch_count = 1
N = 2

MICROBATCH_SIZE = SweepVar(
    *(
        128,
        192,
        256,
        384,
    )
    * 1,
    name="microbatch_size",
)

BATCH_SIZE = MICROBATCH_SIZE * N * device_microbatch_count

intval = Val(value=1)
floatval = Val(value=1.0)
intvar = SweepVar(1, 2, 3, name="var")
NUM_TOKENS_FLOAT = 60e9
NUM_TOKENS = SweepVar(NUM_TOKENS_FLOAT, name="data_scale")

cfg = Cfg(
    a=intval * 1,
    b=intval * 1,
    c=NUM_TOKENS // Val(value=1024) // BATCH_SIZE,
    d=2.0,
)

d = cfg.model_dump_json()

cfg = Cfg.model_validate_json(d)

for i in range(cfg.to_swept_nodes().swept_combinations_count_including_vars()):
    print(cfg.to_swept_nodes().select_instance_by_index(i))
