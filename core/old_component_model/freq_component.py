from nqgl.mlutils.components.cache_layer import ActsCache, CacheProcLayer


from abc import ABC, abstractmethod


class FreqMonitorComponent(ABC):
    @abstractmethod
    def __init__(self, parent: CacheProcLayer = None): ...

    @abstractmethod
    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0): ...

    @abstractmethod
    def get_dead_neurons(self, count_min, threshold): ...

    @property
    @abstractmethod
    def freqs(self): ...

    @classmethod
    def bind_init_args(cls, **kwargs):
        return lambda parent: cls(parent=parent, **kwargs)


class CountingFreqActComponent(FreqMonitorComponent):
    def __init__(self, parent: CacheProcLayer = None):
        self._activations = 0
        self._count = 0
        self._parent = None
        self.parent: CacheProcLayer = parent

    @property
    def parent(self) -> CacheProcLayer:
        return self._parent

    @parent.setter
    def parent(self, parent: CacheProcLayer):
        assert self._parent is None
        self._parent = parent
        if self._parent is not None:
            self._parent.train_cache_template.acts = ...
            self._parent.train_process_after_call.add(self._update_from_cache)
            # self.parent.train_cache_template.register_write_callback("acts", self._update_count_freqs)

    def get_dead_neurons(self, count_min, threshold):
        return super().get_dead_neurons(count_min, threshold)

    def steps_counter(self, cache: ActsCache):
        return cache.acts.shape[0]

    def _get_count_freq(self):
        return self._activations / (self._count + 1e-9)

    def _update_from_cache(self, cache):
        if cache.has.acts:
            self._activations += self.activated_counter(cache)
            self._count += self.steps_counter(cache)

    def activated_counter(self, cache: ActsCache):
        return cache.acts.count_nonzero(dim=0)

    def reset_freqs(self, mask=None, initial_activation=0, initial_count=0):
        if mask is None:
            assert isinstance(self.count, int)
            self.active = torch.zeros_like(self.active) + initial_activation
            self.count = initial_count
        else:
            self.active[mask] = initial_activation
            self.count[mask] = initial_count

    @property
    def freqs(self):
        return self._get_count_freq()


class EMAFreqMixin: ...
