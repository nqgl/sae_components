import torch.nn as nn

from saeco.core.module import Module


class Collection(Module):
    def __init__(
        self,
        *collection_list: nn.Parameter | nn.Module,
        _support_parameters: bool = True,
        _support_modules: bool = True,
        **collection_dict: nn.Parameter | nn.Module,
    ):
        super().__init__()
        assert (len(collection_list) == 0) or (
            len(collection_dict) == 0
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
                    f"Only nn.Modules and nn.Parameters are allowed in collections.\nError@ {name}={module}"
                )

        self._support_parameters = _support_parameters
        self._support_modules = _support_modules

    def __getitem__(self, key):
        if not isinstance(key, slice):
            if isinstance(key, int):
                return getattr(self, self._collection_names[key])
            if key not in self._collection_names:
                raise KeyError(f"{key} not found in {self.__class__.__name__}")
            return getattr(self, key)

        assert key.step is None
        if isinstance(key.start, int) or isinstance(key.stop, int):
            items = list(self._collection.items())[key]
        elif isinstance(key.start, str) or isinstance(key.stop, str):
            numerical_indices = [
                list(self._collection.keys()).index(k) if isinstance(k, str) else k
                for k in (key.start, key.stop)
            ]
            items = list(self._collection.items())[
                numerical_indices[0] : numerical_indices[1]
            ]
        else:
            raise ValueError(
                "key.start and key.stop must be of the same type (int or str)"
            )
        assert "_support_parameters" not in items and "_support_modules" not in items
        return self.__class__(
            _support_parameters=self._support_parameters,
            _support_modules=self._support_modules,
            **{k: v for k, v in items},
        )
