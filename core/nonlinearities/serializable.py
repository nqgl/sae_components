import torch
from typing import Optional


from dataclasses import dataclass
from functools import partial
from typing import Dict

from nqgl.mlutils.components.nonlinearities.undying import undying_relu

nonlinearity_dict = {
    "relu": torch.nn.functional.relu,
    "undying_relu": undying_relu,
    "gelu": torch.nn.functional.gelu,
}


@dataclass
class SerializableNonlinearity:
    nonlinearity: str = "relu"
    nonlinearity_kwargs: Optional[Dict] = None

    def to_nonlinearity(self):
        return partial(
            nonlinearity_dict[self.nonlinearity], **(self.nonlinearity_kwargs or {})
        )

    def __post_init__(self):
        self.nonlinearity_kwargs = self.nonlinearity_kwargs or {}

    def __call__(self, x):
        return self.to_nonlinearity()(x, **self.nonlinearity_kwargs)

    def __eq__(self, other):
        return (
            self.nonlinearity.__name__ == other.nonlinearity.__name__
            and self.nonlinearity_kwargs == other.nonlinearity_kwargs
        )

    @staticmethod
    def ReLU(self):
        return SerializableNonlinearity(nonlinearity="relu")

    @staticmethod
    def UndyingReLU(self, l=0.01, k=1, l_mid_neg=None, l_low_neg=0, l_low_pos=None):
        return SerializableNonlinearity(
            nonlinearity="undying_relu",
            nonlinearity_kwargs={
                "l": l,
                "k": k,
                "l_mid_neg": l_mid_neg,
                "l_low_neg": l_low_neg,
                "l_low_pos": l_low_pos,
            },
        )


def cfg_to_nonlinearity(cfg):
    nonlinearity = nonlinearity_dict[cfg._nonlinearity[0]]
    nonlinearity_kwargs = cfg._nonlinearity[1]
    return partial(nonlinearity, **nonlinearity_kwargs)
