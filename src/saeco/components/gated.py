import saeco.core as cl
import saeco.components as co


class Gated(cl.Parallel):
    def __init__(self, *, gate, values=None):
        super().__init__(
            gate=gate,
            values=values or cl.ops.Identity(),
            _support_parameters=True,
            _support_modules=True,
        )
        self.reduce(lambda g, v: g * v)
