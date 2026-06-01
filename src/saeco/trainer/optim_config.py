from collections.abc import Callable
from enum import StrEnum

import torch
from schedulefree import AdamWScheduleFree
from torch.optim import ASGD, SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSprop, Rprop

_OPTIM_REGISTRY: dict[
    str, type[torch.optim.Optimizer] | Callable[..., torch.optim.Optimizer]
] = {
    "Adam": Adam,
    "RAdam": RAdam,
    "NAdam": NAdam,
    "RMSprop": RMSprop,
    "SGD": SGD,
    "Rprop": Rprop,
    "ASGD": ASGD,
    "AdamW": AdamW,
    "ScheduleFree": AdamWScheduleFree,
    "ScheduleFreeAsNormal": AdamWScheduleFree,
    "AmsgradW": lambda *a, **k: AdamW(*a, **k, amsgrad=True),
    "Adamax": Adamax,
}


def get_optim_cls(
    opt: str,
) -> type[torch.optim.Optimizer] | Callable[..., torch.optim.Optimizer]:
    return _OPTIM_REGISTRY[opt]


class OptimType(StrEnum):
    """Closed set of optimizer selectors accepted by :func:`get_optim_cls`.

    Values mirror the keys of ``_OPTIM_REGISTRY`` (the assertion below keeps
    them in sync), and ``ScheduleFree`` is also the sentinel compared in
    ``TrainConfig.use_schedulefree``. Members are plain strings so persisted
    configs coerce and ``==``/dict-key lookups keep working.
    """

    Adam = "Adam"
    RAdam = "RAdam"
    NAdam = "NAdam"
    RMSprop = "RMSprop"
    SGD = "SGD"
    Rprop = "Rprop"
    ASGD = "ASGD"
    AdamW = "AdamW"
    ScheduleFree = "ScheduleFree"
    ScheduleFreeAsNormal = "ScheduleFreeAsNormal"
    AmsgradW = "AmsgradW"
    Adamax = "Adamax"


assert {t.value for t in OptimType} == set(_OPTIM_REGISTRY), (
    "OptimType members must stay in sync with _OPTIM_REGISTRY keys"
)
