"""saeco: a modular library for training and analyzing sparse autoencoders."""

from saeco.architecture import (
    SAE,
    Architecture,
    ArchitectureBase,
    arch_prop,
    aux_model_prop,
    loss_prop,
    model_prop,
)
from saeco.initializer import InitConfig, Initializer
from saeco.sweeps import (
    SweepableConfig,
    SweepExpression,
    SweepVar,
    Swept,
    Val,
    do_sweep,
)
from saeco.trainer import (
    RunConfig,
    RunSchedulingConfig,
    Trainable,
    TrainConfig,
    Trainer,
)

__all__ = [
    # Architecture & SAE building blocks
    "Architecture",
    "ArchitectureBase",
    "SAE",
    "arch_prop",
    "aux_model_prop",
    "loss_prop",
    "model_prop",
    # Config & sweep DSL
    "InitConfig",
    "RunConfig",
    "RunSchedulingConfig",
    "SweepExpression",
    "SweepVar",
    "Swept",
    "SweepableConfig",
    "TrainConfig",
    "Val",
    # Training
    "Initializer",
    "Trainable",
    "Trainer",
    "do_sweep",
]
