import saeco.core as cl


class L0Targeting(cl.PassThroughModule):
    def __init__(self, target, scale=1.0, increment=0.0003, multiplicative=True):
        super().__init__()
        self.value = scale
        self.target = target
        self.increment = increment
        self.mult = multiplicative

    def process_data(self, x, *, cache):
        l0 = (x != 0).sum(dim=-1).float().mean(0).sum()

        incr = self.increment
        if l0 < self.target:
            incr *= -1
        if self.mult:
            self.value *= 1 + incr
        else:
            self.value += incr

    def get_value(self):
        return self.value
