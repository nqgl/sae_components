from dataclasses import dataclass
from torch.optim import Adam, RAdam, NAdam, RMSprop, SGD, Rprop, ASGD
from saeco.sweeps import SweepableConfig


def get_optim_cls(opt: str):
    return {
        "Adam": Adam,
        "RAdam": RAdam,
        "NAdam": NAdam,
        "RMSprop": RMSprop,
        "SGD": SGD,
        "Rprop": Rprop,
        "ASGD": ASGD,
    }[opt]


class OptimConfig:
    # lr: float = 1e-3
    # betas: tuple[float, float] = (0.9, 0.999)
    name: str = "Adam"

    def optim(self, *args, **kwargs): ...
