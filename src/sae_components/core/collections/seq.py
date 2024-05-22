from typing import Any
import torch.nn as nn
from sae_components.core.cache import Cache
from sae_components.core.module import Module
from sae_components.core.collections.propagator import Propagator


def sequential_rule(x, l, **k):
    if len(l) == 0:
        return x
    return l[-1]


class Seq(Propagator):
    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )
        self.propagate(sequential_rule)
        self.reduce(lambda *l: l[-1])


# TODO: check/test this
def residual_output_rule(out, x, l, **k):
    if len(l) == 0:
        return out + x
    return out + l[-1]


class ResidualSeq(Seq):
    ### I was wrong about the way this can be implemented by propagate rule,
    # it's at least a bit different

    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )
        self._output_rule = residual_output_rule
