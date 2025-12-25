from schedulefree import AdamWScheduleFree
from torch.optim import ASGD, SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSprop, Rprop


def get_optim_cls(opt: str):
    return {
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
    }[opt]
