from sae_components.core.cache import Cache

# from sae_components.core.component_layer.component_layer import ComponentLayer
from sae_components.core.component_layer.require_provide import (
    RequiredReq,
    Req,
)


from abc import ABC, abstractmethod
from typing import List


class LayerComponent(ABC):
    train_cache_watch: List[str] = []
    eval_cache_watch: List[str] = []
    _default_component_name: str = ...
    _requires: List[RequiredReq] = []
    _provides: List[str] = []
    _layer: "ComponentLayer" = None

    @abstractmethod
    def _update_from_cache(self, cache: Cache, **kwargs):
        raise NotImplementedError

    @classmethod
    def bind_nonlayer_args(cls, **kwargs):
        return lambda layer: cls(layer=layer, **kwargs)

    @classmethod
    def bind_named_args(cls, **kwargs):
        return lambda *a, **k: cls(*a, **kwargs, **k)

    def _register_parent_layer(self, layer: "ComponentLayer"):
        self._layer = layer

    def finalize(self):
        for req in self._requires:
            assert (
                req.fulfilled()
            ), f"Requirement {req.req} not fulfilled. Cannot finalize."
