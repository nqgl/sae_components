from typing import Any, Union
from sae_components.core.cache import Cache
from sae_components.core.collections.collection import Collection
from sae_components.core.proc_appropriately import proc_appropriately


class Parallel(Collection):
    def __init__(
        self,
        *,
        binary_reduction=False,
        cache_to_reduction=False,
        support_parameters=True,
        support_modules=True,
        binary_reduction_initial_value=None,
        **collection,
    ):
        super().__init__(
            collection=collection,
            support_parameters=support_parameters,
            support_modules=support_modules,
        )
        self._reduction = None
        self._cache_to_reduction = cache_to_reduction
        self._binary_reduction = binary_reduction
        self._binary_reduction_initial_value = binary_reduction_initial_value
        # ah maybe I should make this take these as list or as dict explicitly not as args and kwargs

    def forward(self, x, *, cache: Cache, **kwargs):
        l = [
            proc_appropriately(module=module, name=name, x=x, cache=cache, **kwargs)
            for name, module in self._collection.items()
        ]
        return self._reduce(*l, cache=cache)

    def _reduce(self, *l, cache):
        if not self._binary_reduction:
            if self._cache_to_reduction:
                return self._reduction(*l, cache=cache)
            return self._reduction(*l)
        if self._binary_reduction_initial_value is None and len(l) <= 1:
            raise ValueError(
                "Binary reduction requires at least 2 elements in the collection"
            )
        if self._binary_reduction_initial_value is not None:
            r = self._reduction(self._binary_reduction_initial_value, l[0])
        else:
            r = l[0]

        for v in l[1:]:
            if self._cache_to_reduction:
                r = self._reduction(r, v, cache=cache)
            else:
                r = self._reduction(r, v)
        return r

    def reduce(self, f, binary=False, takes_cache=False):
        self._reduction = f
        self._binary_reduction = binary
        self._cache_to_reduction = takes_cache
        return self


class AddParallel(Parallel):
    def __init__(self, **collection):
        super().__init__(
            **collection,
            support_modules=True,
            support_parameters=True,
        )
        self.reduce(lambda a, b: a + b, binary=True, takes_cache=False)


class MulParallel(Parallel):
    def __init__(self, **collection):
        super().__init__(
            **collection,
            support_modules=True,
            support_parameters=True,
        )
        self.reduce(lambda a, b: a * b, binary=True, takes_cache=False)
