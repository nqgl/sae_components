from sae_components.core.collections.propagator import Propagator


def router_rule(x, l, **k):
    i = len(l)
    return x[i]


class Router(Propagator):
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
        self.propagate(router_rule)
