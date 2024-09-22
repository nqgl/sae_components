# import passthrough thing use that?
import saeco.core as cl
from saeco.components import Lambda


# class MetricsList:
#     def __init__(self):
#         self.metrics = []
#         self.names = []

#     def add(self, value, cache):
#         self.metrics.append(value)
#         self.names.append(cache._name)

#     def _to_loggable_list(self):
#         return self.metrics


# class GlobalizedCache:
#     def __init__(self, cache, subname=None):
#         self._cache_to_globalize = cache
#         self._subname = subname

#     def __getattribute__(self, name: str):
#         if name in ["_cache_to_globalize", "_write", "_subname"]:
#             return super().__getattribute__(name)
#         return self._cache_to_globalize.__getattribute__(name)

#     def _write(self, name, value):
#         if self._subname is not None:
#             gc: cl.Cache = self._cache_to_globalize._ancestor[self._subname]
#         else:
#             gc: cl.Cache = self._cache_to_globalize._ancestor
#         if gc._watching(name):
#             if not gc._has(name):
#                 gc._write(name, MetricsList())
#             getattr(gc, name).add(value=value, cache=self._cache_to_globalize)
#         return self._cache_to_globalize._write(name, value)

#     def __getitem__(self, name):
#         return self._cache_to_globalize[name]


class GlobalizedCache:
    def __init__(self, cache, subname=None):
        self._cache_to_globalize = cache
        self._subname = subname

    def __getattribute__(self, name: str):
        if name in ["_cache_to_globalize", "_write", "_subname"]:
            return super().__getattribute__(name)
        return self._cache_to_globalize.__getattribute__(name)

    def _write(self, name, value):
        if self._subname is not None:
            self._cache_to_globalize._ancestor[self._subname]._write(name, value)
        else:
            self._cache_to_globalize._ancestor._write(name, value)
        return self._cache_to_globalize._write(name, value)

    def __getitem__(self, name):
        return self._cache_to_globalize[name]


class Metrics(cl.Parallel):
    def __init__(
        self, *, _support_parameters=True, _support_modules=True, **collection_dict
    ):
        if "acts" not in collection_dict:
            collection_dict = {
                **collection_dict,
                "identity": cl.ops.Identity(),
            }
        else:
            assert list(collection_dict.keys())[-1] == "acts"
        super().__init__(
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )

        def metrics_reduce(*l):
            return l[-1]

        super().reduce(metrics_reduce)

    def reduce(self, f, binary=False, takes_cache=False):
        raise RuntimeError("Metrics do not support alternative reductions")


class Metric(Lambda):
    def __init__(self, func):
        super().__init__(func, module=None)  # no wrapping

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        out = super().forward(x, cache=cache, **kwargs)
        assert out.numel() == 1, f"Metric output must be a single value, got {out}"
        return out


class L0(Metric):
    def __init__(self):
        super().__init__(lambda x: (x != 0).sum(-1).float().mean(0).sum())


class L1(Metric):
    def __init__(self):
        super().__init__(lambda x: x.abs().mean(0).sum())


class ActMetrics(Metrics):
    def __init__(self, name=None, globalize_cache=True, **kwargs):
        self.name = name
        d = dict(L1=L1(), L0=L0())
        if "acts" in kwargs:
            raise ValueError("acts is a reserved keyword")
        super().__init__(**d, **kwargs, acts=cl.ops.Identity())

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache.act_metrics_name = ...
        cache.act_metrics_name = self.name
        return super().forward(
            x, cache=GlobalizedCache(cache, subname=self.name), **kwargs
        )
