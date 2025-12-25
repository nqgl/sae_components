from collections.abc import Iterable
from typing import Any, TypeAlias

from pydantic import BaseModel

from saeco.sweeps.sweepable_config.Swept import Swept

CouldHaveSweep: TypeAlias = BaseModel | dict | list


def to_items(target: CouldHaveSweep) -> Iterable[tuple[str, Any]]:
    if isinstance(target, BaseModel):
        return [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    elif isinstance(target, list):
        return map(lambda x: (str(x[0]), x[1]), enumerate(target))
    else:
        assert isinstance(target, dict)
        return target.items()


def has_sweep(target: CouldHaveSweep):
    for name, attr in to_items(target):
        if isinstance(attr, Swept):
            return True
        elif isinstance(attr, CouldHaveSweep):
            if has_sweep(attr):
                return True
    return False


def index_collection(collection: CouldHaveSweep, key: str) -> Any:
    if isinstance(collection, list):
        return collection[int(key)]
    elif isinstance(collection, dict):
        return collection[key]
    else:
        assert isinstance(collection, BaseModel)
        return getattr(collection, key)


def set_collection(collection: CouldHaveSweep, key: str, value: Any) -> Any:
    if isinstance(collection, list):
        collection[int(key)] = value
    elif isinstance(collection, dict):
        collection[key] = value
    else:
        assert isinstance(collection, BaseModel)
        setattr(collection, key, value)
    return collection


def key_in_collection(collection: CouldHaveSweep, key: str) -> bool:
    if isinstance(collection, list):
        return int(key) < len(collection)
    elif isinstance(collection, dict):
        return key in collection
    else:
        assert isinstance(collection, BaseModel)
        return hasattr(collection, key)
