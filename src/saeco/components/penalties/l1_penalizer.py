from .penalty import Penalty
from torch import Tensor


class L1Penalty(Penalty):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def penalty(self, x: Tensor):
        return x.relu().mean(dim=0).sum() * self.scale


class L1PenaltyScaledByDecoderNorm(Penalty):
    def __init__(self, scale=1.0, decoder=None, det_dec_norms=True):
        super().__init__()
        self.scale = scale
        self.decoder = None
        self.det_dec_norms = det_dec_norms
        if decoder is not None:
            self.set_decoder(decoder)

    def penalty(self, x: Tensor):
        mean_act = x.mean(dim=0)
        dec_norms = self.decoder.features[:].norm(dim=1)
        if self.det_dec_norms:
            dec_norms = dec_norms.detach()
        scaled_acts = dec_norms * mean_act
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
