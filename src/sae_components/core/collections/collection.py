# from sae_components.core.reused_forward import ReuseForward
from typing import Any, Union


from sae_components.core.module import Module


import torch.nn as nn


class Collection(Module):
    def __init__(
        self,
        *collection_list: Union[nn.Parameter, nn.Module],
        _support_parameters=True,
        _support_modules=True,
        **collection_dict: Union[nn.Parameter, nn.Module],
    ):
        super().__init__()
        assert (len(collection_list) > 0) ^ (
            len(collection_dict) > 0
        ), "Either unnamed or named modules should be provided, but not both"

        if len(collection_list) > 0:
            d = {"item" + str(i): module for i, module in enumerate(collection_list)}
        else:
            d = collection_dict

        self._collection_names = list(d.keys())
        self._collection = d  # kind of is useless but maybe this is good to have
        for name, module in d.items():
            if hasattr(self, name):
                raise AttributeError(
                    f"Attribute {name} already exists in {self.__class__.__name__}"
                )
            if isinstance(module, nn.Module):
                if not _support_modules:
                    raise ValueError(
                        "This collection type does not support modules, but module provided"
                    )
                self.add_module(name, module)
            elif isinstance(module, nn.Parameter):
                if not _support_parameters:
                    raise ValueError(
                        "This collection type does not support parameters, but parameter provided"
                    )
                self.register_parameter(name, module)
            else:
                raise ValueError(
                    "Only nn.Modules and nn.Parameters are allowed in collections"
                )

    def __getitem__(self, key):
        if isinstance(key, int):
            return getattr(self, self._collection_names[key])
        if key not in self._collection_names:
            raise KeyError(f"{key} not found in {self.__class__.__name__}")
        return getattr(self, key)

    # def __getattr__(self, key):
    #     if key in super().__getattr__("_collection"):
    #         return super().__getattr__("_collection")[key]
    #     return super().__getattr__(key)
