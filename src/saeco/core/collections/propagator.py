from typing import Any, Literal, Union, Callable, overload
from typing_extensions import Self
from saeco.core.cache import Cache
from saeco.core.collections.collection import Collection
from saeco.core.proc_appropriately import proc_appropriately
from typing import Protocol, runtime_checkable
from torch import Tensor
from types import FunctionType, LambdaType


@runtime_checkable
class PropagateRule(Protocol):
    def __call__(self, x: Tensor, l: list[Tensor], **k) -> Tensor: ...
@runtime_checkable
class OutputRule(Protocol):
    def __call__(self, out: Tensor, x: Tensor, l: list[Tensor], **k) -> Tensor: ...


def output_normal(out, x, l, **k):
    return out


class Propagator(Collection):
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
        self._reduction = None
        self._cache_to_reduction = None
        self._binary_reduction = None
        self._binary_reduction_initial_value = None
        self._propagate_rule: PropagateRule = None
        self._output_rule = output_normal
        # ah maybe I should make this take these as list or as dict explicitly not as args and kwargs

    def propagate(self, propagate_rule: PropagateRule):
        if not isinstance(propagate_rule, PropagateRule):
            raise ValueError(
                "propagate_rule must be a callable with the correct signature"
            )
        self._propagate_rule = propagate_rule
        return self

    def _propagate(self, x, l, *a, **k):
        # decision: assuming no variation in first layer case desired behavior
        return self._propagate_rule(x=x, l=l, *a, **k)

    def forward(self, x, *, cache: Cache, **kwargs):
        l = []
        for name in self._collection_names:
            module = getattr(self, name)
            next_in = self._propagate(x, l, **kwargs)
            l.append(
                self._output_rule(
                    out=proc_appropriately(
                        module=module, name=name, x=next_in, cache=cache, **kwargs
                    ),
                    x=x,
                    l=l,
                    **kwargs,
                )
            )
        return self._reduce(*l, cache=cache)

    def _reduce(self, *l, cache):
        if not self._binary_reduction:
            if self._cache_to_reduction:
                return self._reduction(*l, cache=cache)
            return self._reduction(*l)
        if (self._binary_reduction_initial_value is not None) + len(l) <= 1:
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

    # @overload
    # def reduce(
    #     self,
    #     f: Callable[..., Any],
    #     binary: Literal[False],
    #     takes_cache: bool = False,
    #     binary_initial_value: Any | None = None,
    # ) -> Self: ...
    # @overload
    # def reduce(
    #     self,
    #     f: Callable[[Tensor, Tensor], Tensor],
    #     binary: Literal[True],
    #     takes_cache: bool = False,
    #     binary_initial_value: Any | None = None,
    # ) -> Self: ...

    def reduce(
        self,
        f: FunctionType | Callable[..., Any],
        binary: bool = False,
        takes_cache: bool = False,
        binary_initial_value: Any | None = None,
    ) -> Self:
        self._reduction = f
        self._binary_reduction = binary
        self._binary_reduction_initial_value = binary_initial_value
        self._cache_to_reduction = takes_cache
        return self

    def then(self, *args, **kwargs): ...
