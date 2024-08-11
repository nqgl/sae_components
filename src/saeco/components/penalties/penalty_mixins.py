import torch


class ActsScaledByDecoderNormPenaltyMixin:
    def __init__(self, decoder=None, det_dec_norms=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = None
        self.det_dec_norms = det_dec_norms
        if decoder is not None:
            self.set_decoder(decoder)

    def penalty(self, x, *, cache):
        # x: b, f
        dec_norms = self.decoder.features[:].norm(dim=1)
        if self.det_dec_norms:
            dec_norms = dec_norms.detach()
        return super().penalty(x * dec_norms.unsqueeze(0), cache=cache)

    def set_decoder(self, decoder):
        assert self.decoder is None
        self.decoder = decoder.features["weight"]
        return decoder


class LinearDecayPenaltyMixin:
    def __init__(self, end, begin=0, end_scale=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            scale = 1
        else:
            prog = (self.end - step) / (self.end - self.begin)
            scale = prog + self.end_scale * (1 - prog)
        return super().penalty(x, cache=cache) * scale
