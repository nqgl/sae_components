import torch

from saeco.core.collections.propagator import Propagator


def sequential_rule(x, l, **k):
    if len(l) == 0:
        return x
    return l[-1]


class Seq(Propagator):
    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )
        self.propagate(sequential_rule)
        self.reduce(lambda *l: l[-1])


# TODO: check/test this
def residual_output_rule(out, x, l, **k):
    if len(l) == 0:
        return out + x
    return out + l[-1]


class ResidualSeq(Seq):
    ### I was wrong about the way this can be implemented by propagate rule,
    # it's at least a bit different
    # so I switched to the output rule

    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )
        self._output_rule = residual_output_rule


def catseq_output_rule(out, x, l, **k):
    return torch.cat([x, out], dim=-1)


class CatSeq(Propagator):
    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )

        self._output_rule = catseq_output_rule
        self.propagate(sequential_rule)
        self.reduce(lambda *l: l[-1])


def resid_catseq_output_rule(out, x, l, **k):
    if len(l) == 0:
        return torch.cat([x, out], dim=-1)
    return torch.cat([x, out + l[-1][..., x.shape[-1] :]], dim=-1)


class CatSeqResid(Propagator):
    def __init__(
        self,
        *collection_list,
        _support_parameters=False,
        _support_modules=True,
        **collection_dict,
    ):
        super().__init__(
            *collection_list,
            _support_parameters=_support_parameters,
            _support_modules=_support_modules,
            **collection_dict,
        )

        self._output_rule = catseq_output_rule
        self.propagate(sequential_rule)
        self.reduce(lambda *l: l[-1])
