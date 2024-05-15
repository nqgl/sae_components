from nqgl.mlutils.components.cache_layer import CacheLayer, CacheProcLayer
from typing import Dict, List, Union
import torch.nn as nn
from unpythonic import box

from nqgl.mlutils.components.component_layer.layercomponent import LayerComponent


def provides_field(
    obj,
): ...


def islambda(f):
    return callable(f) and getattr(f, "__name__", "") == "<lambda>"


class ComponentLayer(CacheProcLayer):
    def __init__(
        self,
        cachelayer: CacheLayer,
        components: List[LayerComponent] = [],
        names: Dict[LayerComponent, str] = {},
        train_cache=None,
        eval_cache=None,
    ):
        super().__init__(cachelayer, train_cache=train_cache, eval_cache=eval_cache)
        components = [c(self) if islambda(c) else c for c in components]
        self.components = components
        self.module_components = nn.ModuleList(
            [c for c in components if isinstance(c, nn.Module)]
        )
        self._init_register_components(components, names)

    def _init_register_components(self, components, names):
        self._init_update_watched(components)
        self._init_update_attrs_from_components(components, names)
        for c in components:
            if hasattr(c, "_register_parent_layer"):
                c._register_parent_layer(self)

    def _init_update_watched(self, components):
        for c in components:
            if c.train_cache_watch:
                for name in c.train_cache_watch:
                    self.train_cache_template._watch(name)
            if c.eval_cache_watch:
                for name in c.eval_cache_watch:
                    self.eval_cache_template._watch(name)

    def _init_update_attrs_from_components(self, components, names):
        for c in components:
            if c in names:
                name = names[c]
            else:
                name = c._default_component_name
            if name is None:
                continue
            if hasattr(self, name):
                raise ValueError(f"Component name {name} already exists")
            setattr(self, name, c)

    def _update(self, cache, **kwargs):
        super()._update(cache, **kwargs)
        for c in self.components:
            c._update_from_cache(cache=cache, training=self.training, **kwargs)

    def component_architectures(self):
        components_mros = [c.__class__.__mro__ for c in self.components]
        return [
            [str(x).split("'")[1].split(".")[-1] for x in component_mro]
            for component_mro in components_mros
        ]
