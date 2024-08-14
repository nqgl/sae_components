from saeco.architectures.gate_hierarch import (
    hierarchical_softaux,
    HierarchicalSoftAuxConfig,
)
from saeco.trainer.RunConfig import RunConfig
from saeco.trainer.TrainConfig import TrainConfig
from saeco.trainer.runner import TrainingRunner
from saeco.sweeps import Swept, SweepableConfig

PROJECT = "sweep test"


class TestConfig(SweepableConfig):
    a_str: str
    an_int: int = 1


class NestConfig(SweepableConfig):
    tcfg: TestConfig
    an_int: int = 1


cfg = NestConfig(
    tcfg=Swept[TestConfig](
        TestConfig(
            a_str=Swept[str]("a", "b"),
            an_int=Swept[int](12),
        )
    ),
    an_int=Swept[int](1),
)


def run(cfg):
    print(cfg)
