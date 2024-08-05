from .penalty import Penalty
from torch import Tensor
import torch


class L1Penalty(Penalty):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def penalty(self, x, *, cache):
        return x.abs().mean(dim=0).sum() * self.scale


class LinearDecayL1Penalty(Penalty):
    def __init__(self, end, begin=0, scale=1.0, end_scale=0):
        super().__init__()
        self.scale = scale
        self.begin = begin
        self.end = end
        self.end_scale = end_scale

    def penalty(self, x, *, cache):
        if not cache._ancestor.has.trainstep:
            return torch.zeros(1)
        step = cache._ancestor.trainstep
        if step > self.end:
            if self.end_scale == 0:
                return torch.zeros(1)
            scale = self.end_scale
        elif step <= self.begin:
            scale = self.scale
        else:
            prog = (self.end - step) / (self.end - self.begin)
            scale = self.scale * prog + self.end_scale * (1 - prog)
        return x.abs().mean(dim=0).sum() * scale


class L1PenaltyScaledByDecoderNorm(Penalty):
    def __init__(self, scale=1.0, decoder=None, det_dec_norms=False):
        super().__init__()
        self.scale = scale
        self.decoder = None
        self.det_dec_norms = det_dec_norms
        if decoder is not None:
            self.set_decoder(decoder)

    def penalty(self, x, *, cache):
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

    def penalty(self, x, *, cache):
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
