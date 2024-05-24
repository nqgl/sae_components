# import passthrough thing use that?
import sae_components.core as cl


class Metrics(cl.Parallel):
    def __init__(
        self, *, _support_parameters=True, _support_modules=True, **collection_dict
    ):
        collection_dict = {"identity": cl.ops.Identity(), **collection_dict}
        super().__init__(
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict
        )
        super().reduce(lambda *l: l[0])

    def reduce(self, f, binary=False, takes_cache=False):
        raise RuntimeError("Metrics do not support alternative reductions")


# class ActMetrics()
