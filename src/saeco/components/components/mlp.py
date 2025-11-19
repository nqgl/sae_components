import torch.nn as nn

import saeco.core as cl


class MLP(cl.Seq):
    def __init__(self, *, dims: list[int], nonlinearity: nn.Module, bias=True):
        assert len(dims) > 2
        super().__init__(
            *[
                item
                for pair in [
                    (nn.Linear(in_features, out_features, bias=bias), nonlinearity)
                    for in_features, out_features in zip(dims, dims[1:])
                ]
                for item in pair
            ][:-1]
        )


class FeedForward(MLP):
    def __init__(
        self,
        d_in,
        d_out=None,
        d_hidden=None,
        expansion_factor=None,
        nonlinearity=nn.ReLU,
        bias=True,
    ):
        assert (d_hidden is None) != (expansion_factor is None)
        if d_hidden is None:
            d_hidden = d_in * expansion_factor
        if d_out is None:
            d_out = d_in
        super().__init__(
            dims=[d_in, d_hidden, d_out], nonlinearity=nonlinearity, bias=bias
        )
