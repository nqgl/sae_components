from typing import Any, List, TypeVar
from dataclasses import dataclass, field, Field
from jaxtyping import Float, jaxtyped
from torch import Tensor, NumberType
from saeco.misc.exception_location_hint import locate_cache_exception

import inspect
import re

# @dataclass

T = TypeVar("T")


class CacheHas:
    def __init__(self, cache):
        self._cache: Cache = cache

    def __getattribute__(self, _name: str) -> Any:
        if _name.startswith("_"):
            return super().__getattribute__(_name)
        return self._cache._has(_name)


class CacheWatching:
    def __init__(self, cache):
        self._cache: Cache = cache

    def __getattribute__(self, _name: str) -> Any:
        if _name.startswith("_"):
            return super().__getattribute__(_name)
        return self._cache._watching(_name)


def cancopy(v):
    return hasattr(v, "copy")


def listdictcopy(dl):
    return {k: (v.copy() if cancopy(v) else v) for k, v in dl.items()}


def listdictadd(da, db, unique=True):
    do = listdictcopy(da)
    for k, v in db.items():
        if k in da:
            summed = do[k] + v
            do[k] = list(dict.fromkeys(summed)) if unique else summed
        else:
            do[k] = v.copy() if cancopy(v) else v
    return do


def dlcopy(dl):
    if isinstance(dl, list):
        return dl.copy()
        # return [dlcopy(i) for i in dl]
    if isinstance(dl, dict):
        return {k: dlcopy(v) for k, v in dl.items()}
    if cancopy(dl):
        return dl.copy()
    return dl


def dlmerge(da, db, unique=True):
    da = dlcopy(da)
    for k, vb in db.items():
        if k in da:
            assert type(da[k]) == type(
                vb
            ), f"Type mismatch: {type(da[k])} and {type(vb)}"
            if isinstance(vb, list):
                summed = da[k] + vb
                da[k] = list(dict.fromkeys(summed)) if unique else summed
            elif isinstance(vb, dict):
                da[k] = dlmerge(da[k], vb, unique=unique)
            else:
                raise Exception(f"dlmerge Cannot merge type {type(vb)}")
        else:
            da[k] = dlcopy(vb)
    return da


"""
A different way this could work, is like:
cache.watch.attributenametowatch = True/False
yeah probably I will rewrite this to use Dict[str, Bool]


also possibly todo if this seems useful:
cache[i] -> cache (
    subcache of first cache, default to watch same attributes 
    (make as cache.clone() probably)
)

Cache c:
ci:Cache = c[i]
ci._parent == c
ci._prev = c[i - 1] if i > 0 else None



sibling cache way to access siblings and have it fail if you have the wrong key
instead of doing prev_cache or .parent[self.key-1]
"""


class Cache:
    """
    Fields are write-once.
    Default value to denote a watched attribute (__NULL_ATTR) is ellipsis
    """

    class NullType:
        watched: bool

    class NotWatched: ...

    class Expected:
        def __init__(self):
            raise NotImplementedError

        in_family: bool
        in_ancestors: bool
        in_descendants: bool
        in_self: bool
        in_siblings: bool
        anywhere_if_watched: bool
        each_if_watched: bool
        exclude_from_log: bool

        class InFamilty: ...

    class ExpectedIfWatched:
        def __init__(self):
            raise NotImplementedError

    __RESERVED_NAMES = ["has", "watching", "destruct"]
    _NULL_ATTR = ...
    _NULLTYPES = [NotWatched, Expected, ExpectedIfWatched, _NULL_ATTR]  # TODO these
    _unwatched_writes = ...
    _ignored_names = ...
    _write_callbacks = ...
    _lazy_read_funcs = ...
    _name: str = ...
    has: CacheHas  # TypeVar this so it knows it's contents would be nice. maybe possible?
    watching: CacheWatching

    def __init__(self, callbacks=None, parent=None, subcache_index=None):
        self._unwatched_writes: set = set()
        self._ignored_names: set = set()
        self._write_callbacks: dict = callbacks or {}
        self._lazy_read_funcs: dict = {}
        self._subcaches: dict = {}
        self._parent: Cache = parent
        self._subcache_index = subcache_index
        self._is_root = parent is None
        super().__setattr__("has", CacheHas(self))
        super().__setattr__("watching", CacheWatching(self))

    @property
    def _name(self):
        if self._subcache_index is None:
            return "root"
        n = self._subcache_index
        return f"{self._parent._name}.{n}"

    def _watch(self, _name: str | list[str]):
        if isinstance(_name, list):
            for name in _name:
                self._watch(name)
            return self
        if _name in self._lazy_read_funcs:
            raise AttributeError(f"Attribute {_name} is lazy-rendered")
        self.__setattr__(_name, ...)
        return self

    def _watching(self, _name: str):
        return hasattr(self, _name) and not self._ignored(_name)
        # raise NotImplementedError("Not yet implemented")

    def __setattr__(self, _name: str, __value: Any) -> None:
        if _name in self.__RESERVED_NAMES:
            raise AttributeError(f"Cannot set reserved attribute {_name}")

        if _name.startswith("_"):
            return super().__setattr__(_name, __value)

        if __value == self._NULL_ATTR:
            if hasattr(self, _name) and not getattr(self, _name) in self._NULLTYPES:
                raise AttributeError(
                    f"Cache error: Tried to watch attribute {_name}, but {_name} already set to {getattr(self, _name)}"
                )
            else:
                return super().__setattr__(_name, __value)
        self._write(_name, __value)

    def _has(self, _name: str):
        return (
            hasattr(self, _name) and super().__getattribute__(_name) != self._NULL_ATTR
        ) or _name in self._lazy_read_funcs

    def _ignored(self, _name: str):
        return _name in self._ignored_names or not hasattr(self, _name)

    def _write(self, _name: str, __value: Any):
        if self._ignored(_name):
            if _name in self._unwatched_writes:
                raise AttributeError(
                    f"Cache overwrite error on unwatched attribute: Unwatched attribute {_name} already written"
                )
            self._unwatched_writes.add(_name)
        elif getattr(self, _name) != self._NULL_ATTR:
            raise AttributeError(
                f"Cache overwrite error: Watched attribute {_name} already set to {getattr(self, _name)}"
            )
        if _name in self._write_callbacks:
            for nice in sorted(self._write_callbacks[_name].keys()):
                for hook in self._write_callbacks[_name][nice]:
                    o = hook(self, __value)
                    if o is not None:
                        __value = o
        if self._watching(_name):
            super().__setattr__(_name, __value)
        return __value

    def _getfields(self):
        values = {}
        watching = set()
        names = {
            name
            for name in self.__dict__
            if not ((name.startswith("_") or name in self.__RESERVED_NAMES))
        }  # - set(self.__class__.__dict__.keys()) TODO was I correct to remove this?
        for name in names:
            if self._has(name):
                values[name] = getattr(self, name)
            if self._watching(name):
                watching.add(name)
        return watching, values

    def __iadd__(self, other: "Cache"):
        if not isinstance(other, Cache):
            raise TypeError(
                f"Cannot add {other.__class__} to Cache. Must be Cache or subclass"
            )
        o_watching, o_values = other._getfields()
        for watch in o_watching:
            if not self._watching(watch):
                self.__setattr__(watch, ...)
        for name, value in o_values.items():
            self.__setattr__(name, value)
        self._write_callbacks = dlmerge(self._write_callbacks, other._write_callbacks)
        self._unwatched_writes = self._unwatched_writes.union(other._unwatched_writes)
        self._ignored_names = self._ignored_names.union(other._ignored_names)
        self._lazy_read_funcs = dlmerge(self._lazy_read_funcs, other._lazy_read_funcs)
        if other._parent is not None and self._parent is not other:
            pass
            # raise NotImplementedError("cache copy receiving _parent not yet supported")
        if not other._subcaches == {} and self._parent is not other:
            raise NotImplementedError(
                "cache copy recieving _subcaches not yet supported"
            )
        assert self._NULL_ATTR == other._NULL_ATTR
        return self

    def parent_iadd(self, other: "Cache"):
        if not isinstance(other, Cache):
            raise TypeError(
                f"Cannot add {other.__class__} to Cache. Must be Cache or subclass"
            )
        o_watching, o_values = other._getfields()
        for watch in o_watching:
            if not self._watching(watch):
                self.__setattr__(watch, ...)
        # Commenting the below out so we just copy watched status

        # for name, value in o_values.items():
        #     self.__setattr__(name, value)
        self._write_callbacks = dlmerge(self._write_callbacks, other._write_callbacks)
        # self._unwatched_writes = self._unwatched_writes.union(other._unwatched_writes)
        self._ignored_names = self._ignored_names.union(other._ignored_names)
        self._lazy_read_funcs = dlmerge(
            self._lazy_read_funcs, other._lazy_read_funcs
        )  # Maybe should rm this too or make it more explicit
        if other._parent is not None and self._parent is not other:
            raise NotImplementedError("cache copy receiving _parent not yet supported")
        if not other._subcaches == {} and self._parent is not other:
            raise NotImplementedError(
                "cache copy recieving _subcaches not yet supported"
            )
        assert self._NULL_ATTR == other._NULL_ATTR
        return self

    def __getitem__(self, i):
        if i in self._subcaches:
            return self._subcaches[i]
        else:
            subcache = self.clone(parent=True)
            subcache._subcache_index = i
            self._subcaches[i] = subcache
            return subcache

    @property
    def _prev_cache(self):
        if self._parent is None:
            raise AttributeError("No parent cache")
        index = self._subcache_index - 1
        if index not in self._parent._subcaches:
            raise AttributeError("No previous cache")
        return self._parent[index]

    def register_write_callback(self, _name: str, hook, ignore=False, nice=0):
        """
        hook fn: (cache, value) -> Optional[value]
            if hook returns a value, that value will replace the input value as what gets written into the cache
        """
        if _name.startswith("_"):
            raise AttributeError("Cannot set hook on private attribute")
        if _name not in self._write_callbacks:
            self._write_callbacks[_name] = {}
        if nice not in self._write_callbacks[_name]:
            self._write_callbacks[_name][nice] = []
        self._write_callbacks[_name][nice].append(hook)
        if ignore:
            self.add_cache_ignore(_name)

    def add_cache_ignore(self, _name: str):
        if _name.startswith("_"):
            raise AttributeError("Cannot ignore private attribute")
        self._ignored_names.add(_name)
        return self

    def clone(self, parent=False):
        if parent:
            clone = self.__class__(parent=self)
            # clone._parent = self
            clone.parent_iadd(self)
        else:
            clone = self.__class__()
            clone += self
        return clone

    # def __iter__ ?
    #

    def logdict(
        self,
        name="cache",
        excluded: List[str] = [],
        exclude_contains: List[str] = [],
        itemize=True,
    ):
        _, vals = self._getfields()
        for ex in excluded:
            if ex in vals:
                vals.pop(ex)
        if exclude_contains:
            exclude_query = "|".join(exclude_contains)
            for p in filter(
                lambda x: re.search(exclude_query, x) is not None, list(vals.keys())
            ):
                vals.pop(p)

        values = {}
        for k, v in vals.items():
            if hasattr(v, "_to_loggable_form"):
                v = v._to_loggable_list()
                if len(v) == 1:
                    v = v[0]
                else:
                    v = [
                        i
                        for i in v
                        if isinstance(i, Tensor)
                        and itemize
                        and i.numel() == 1
                        or isinstance(i, NumberType)
                    ]
                    values[k] = v
                    continue
            if (
                isinstance(v, Tensor)
                and itemize
                and v.numel() == 1
                or isinstance(v, NumberType)
            ):
                values[k] = v.item()
            elif isinstance(v, int) or isinstance(v, float):
                values[k] = v
        subcaches = [
            v.logdict(name=f"{name}/{k}", excluded=excluded, itemize=itemize)
            for k, v in self._subcaches.items()
        ]
        cache = {f"{name}/{k}": v for k, v in values.items()}
        for subcache_dict in subcaches:
            cache.update(subcache_dict)
        return cache

    def fullkey(self):
        """
        returns a tuple of all the keys to get from parent to here
        """
        raise NotImplementedError

    def search(self, attr):
        """
        returns all sub(or self)caches that have attr
        usage
        ```
        l = []
        for c in cache.search("a"):
            l.append(c, c.a)
        for c in cache.search("b"):
            print(c.fullkey())
        ```
        """
        l = []
        nex = [self]
        while nex:
            cur = nex
            nex = []
            for c in cur:
                if c._has(attr):
                    l.append(c)
                nex.extend(c._subcaches.values())
        return l

    def find_singular(self, attr: str):
        l = self.search(attr)
        if len(l) != 1:
            raise ValueError(
                f"found {len(l)} caches with attr {attr}. expected exactly one."
            )
        return getattr(l[0], attr)

    # def _search_children(self, attr):
    #     l = []
    #     for k, v in self._subcaches.items():
    #         if v._has(attr):
    #             l.append(v)
    #     return l

    @property
    def _ancestor(self):
        a = self
        while a._parent is not None:
            a = a._parent
        return a

    def destruct(self):
        for k, cache in [i for i in self._subcaches.items()]:
            cache.destruct()
            del self._subcaches[k]
            del cache

        del self.__dict__

    def __del__(self):
        if hasattr(self, "_is_root") and self._is_root:
            self.destruct()

    @property
    def _is_dead(self):
        return len(self.__dict__) == 0

    def __call__(self, obj: T, force_watch=False) -> "SubCacher | T":
        # the typing is not technically correct here
        # but it gets the fields right in the IDE
        return SubCacher(cache=self, obj=obj, force_watch=force_watch)


class SubCacher:
    def __init__(self, cache, obj, record=False, force_watch=None):
        self._cache: Cache = cache
        self._obj = obj
        self._record = record or force_watch
        self._force_watch = force_watch
        # assert not force_watch or record  # force_watch -> record

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        obj = getattr(self._obj, name)
        subcache = self._cache[name]
        return SubCacher(
            cache=subcache, obj=obj, record=self._record, force_watch=self._force_watch
        )

    def __call__(self, *args, **kwargs):
        assert "cache" not in kwargs
        if (
            "cache" in inspect.signature(self._obj).parameters
            or hasattr(self._obj, "forward")
            and "cache" in inspect.signature(self._obj.forward).parameters
        ):
            kwargs["cache"] = self._cache
        try:
            v = self._obj(*args, **kwargs)
        except Exception as e:
            raise locate_cache_exception(e, self._cache)
        if self._record:
            record_location = self._cache._subcache_index
            if isinstance(self._record, str):
                record_location += self._record
            if self._force_watch:
                self._cache._parent._watch(record_location)
            return self._cache._parent._write(record_location, v)
        return v

    def __getitem__(self, i):
        return SubCacher(
            cache=self._cache[i],
            obj=self._obj[i],
            record=self._record,
            force_watch=self._force_watch,
        )


class CacheSpec(Cache):
    def __init__(self):
        raise Exception(
            "Do not instantiate a CacheSpec, use it for interface hinting only"
        )


def main():
    class TC(Cache):
        tf = ...

    c = Cache()
    tc = TC()
    tc2 = TC()
    import torch

    tc.tf = 3
    tc2.tf = 5
    # TC.tf = 3
    print(tc.tf)
    print(tc2.tf)
    tc.tf = 4
    print(tc.tf)

    # type_example = Float[Tensor, "b c"]
    # # t = type_example(torch.rand(3, 4))
    # t(torch.rand(3, 4), torch.rand(3, 4))
    # t(torch.rand(3, 4), torch.rand(3, 5))

    # t(torch.rand(3, 4), torch.rand(2, 5))

    # print(type_example)


if __name__ == "__main__":
    main()


def atest(c: Cache):
    l = []
    l.append(repr(c))
    c2 = c.clone()
    l.append(repr(c))
    l.append(repr(c2))
    c += c2
    l.append(repr(c))
    c += c
    l.append(repr(c))
