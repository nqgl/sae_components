import einops

import saeco.core as cl
from saeco.components.threshgate import RandSoftThreshGate


class GotActsInterrupt(Exception):
    def __init__(self, acts):
        super().__init__()
        self.acts = acts


def cache_acts_interrupt_hook(cache: cl.Cache, acts):
    cache.layer_acts = acts
    raise GotActsInterrupt(acts)


def get_acts_only(model, x, cache: cl.Cache, actsname="acts", **kwargs):
    cache = cache.clone()
    cache.register_write_callback(actsname, cache_acts_interrupt_hook)
    try:
        model(x, cache=cache, **kwargs)
    except GotActsInterrupt as e:
        return e.acts
    raise ValueError("Model did not set acts")


def hl2ll(hl, bf):
    return einops.repeat(hl, "b i -> b (i bf)", bf=bf)


def gate_ll_acts(acts, hl_acts, bf, threshgate):
    if hl_acts is None:
        return acts
    return threshgate(acts, hl2ll(hl_acts, bf))


class CacheProcessor(cl.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        cache = self.preprocess_cache(cache, **kwargs)
        out = cache(self).model(x)
        self.postprocess_cache(cache, **kwargs)
        return out

    def preprocess_cache(self, cache: cl.Cache, **kwargs): ...

    def postprocess_cache(self, cache: cl.Cache, **kwargs): ...


class ActsGateSubstitutor(CacheProcessor):
    def __init__(
        self, model, ll_acts_key, bf, hl_acts_key="prev_acts", threshgate=None
    ):
        (super().__init__(model),)
        self.ll_acts_key = ll_acts_key
        self.hl_acts_key = hl_acts_key
        self.bf = bf
        self.threshgate = threshgate or RandSoftThreshGate(bf=bf)

    def preprocess_cache(self, cache: cl.Cache, **kwargs):
        def process_acts(subcache, acts):
            if kwargs[self.hl_acts_key] is None:
                cache.gated_acts = acts
                return None
            cache.natural_acts = acts
            gated_acts = gate_ll_acts(
                acts, kwargs[self.hl_acts_key], bf=self.bf, threshgate=self.threshgate
            )
            cache.gated_acts = gated_acts
            return gated_acts

        cache.register_write_callback(self.ll_acts_key, process_acts, nice=-2)
        return cache
