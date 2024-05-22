# from sae_components.core.reused_forward import ReuseForward
from typing import Any, Union


from sae_components.core.module import Module


import torch.nn as nn


class Collection(Module):
    def __init__(
        self,
        collection: Union[dict[nn.Parameter, nn.Module], list[nn.Parameter, nn.Module]],
        support_parameters=True,
        support_modules=True,
    ):
        super().__init__()
        if isinstance(collection, list):
            d = {"item" + str(i): module for i, module in enumerate(collection)}
        else:
            d = collection

        self._collection_names = list(d.keys())
        self._collection = d  # kind of is useless but maybe this is good to have
        for name, module in d.items():
            if hasattr(self, name):
                raise AttributeError(
                    f"Attribute {name} already exists in {self.__class__.__name__}"
                )
            if isinstance(module, Module):
                if not support_modules:
                    raise ValueError(
                        "This collection type does not support modules, but module provided"
                    )
                self.add_module(name, module)
            elif isinstance(module, nn.Parameter):
                if not support_parameters:
                    raise ValueError(
                        "This collection type does not support parameters, but parameter provided"
                    )
                self.register_parameter(name, module)

    def __getitem__(self, key):
        if isinstance(key, int):
            return getattr(self, self._collection_names[key])
        if key not in self._collection_names:
            raise KeyError(f"{key} not found in {self.__class__.__name__}")
        return getattr(self, key)

    def __getattr__(self, key):
        if key in super().__getattr__("_collection"):
            return super().__getattr__("_collection")[key]
        return super().__getattr__(key)
