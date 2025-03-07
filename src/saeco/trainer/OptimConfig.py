from torch.optim import Adam, RAdam, NAdam, RMSprop, SGD, Rprop, ASGD
from schedulefree import AdamWScheduleFree


def get_optim_cls(opt: str):
    return {
        "Adam": Adam,
        "RAdam": RAdam,
        "NAdam": NAdam,
        "RMSprop": RMSprop,
        "SGD": SGD,
        "Rprop": Rprop,
        "ASGD": ASGD,
        "ScheduleFree": AdamWScheduleFree,
        "ScheduleFreeAsNormal": AdamWScheduleFree,
    }[opt]
