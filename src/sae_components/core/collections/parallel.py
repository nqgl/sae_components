from typing import Any, Union
from sae_components.core.cache import Cache
from sae_components.core.collections.collection import Collection
from sae_components.core.proc_appropriately import proc_appropriately
from sae_components.core.collections.propagator import Propagator


def parallel_rule(x, l, **k):
    return x


class Parallel(Propagator):
    def __init__(
        self,
        *collection_list,
        _support_parameters=True,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )
        self.propagate(parallel_rule)


class AddParallel(Parallel):
    def __init__(self, **collection):
        super().__init__(
            _support_modules=True,
            _support_parameters=True,
            **collection,
        )
        self.reduce(lambda a, b: a + b, binary=True, takes_cache=False)


class MulParallel(Parallel):
    def __init__(self, **collection):
        super().__init__(
            _support_modules=True,
            _support_parameters=True,
            **collection,
        )
        self.reduce(lambda a, b: a * b, binary=True, takes_cache=False)
