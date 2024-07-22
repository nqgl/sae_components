from .penalty import Penalty
from torch import Tensor


class L1Penalty(Penalty):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def penalty(self, x: Tensor):
        return x.relu().mean(dim=0).sum() * self.scale


class L1PenaltyScaledByDecoderNorm(Penalty):
    def __init__(self, scale=1.0, decoder=None):
        super().__init__()
        self.scale = scale
        self.decoder = None
        if decoder is not None:
            self.set_decoder(decoder)

    def penalty(self, x: Tensor):
        mean_act = x.mean(dim=0)
        scaled_acts = self.decoder.features[:].norm(dim=1) * mean_act
        return scaled_acts.sum() * self.scale

    def set_decoder(self, decoder):
        assert self.decoder is None
        self.decoder = decoder.features["weight"]
        return decoder


class L0TargetingL1Penalty(Penalty):
    def __init__(self, target, scale=1.0):
        super().__init__()
        self.scale = scale
        self.target = target
        self.increment = 0.00003

    def penalty(self, x: Tensor):
        return x.relu().mean(dim=0).sum() * self.scale

    def update_l0(self, x: Tensor):
        if self.target is None:
            return x
        l0 = (x > 0).sum(dim=-1).float().mean(0).sum()
        if l0 > self.target:
            self.scale *= 1 + self.increment
        else:
            self.scale *= 1 - self.increment
        return x
