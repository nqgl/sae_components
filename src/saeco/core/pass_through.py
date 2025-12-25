# @runtime_checkable
# class FreqTracker(Protocol):
#     def update(self, acts: torch.Tensor): ...
#     @property
#     def freqs(self) -> torch.Tensor: ...
#     def reset(self): ...


from abc import abstractmethod

from saeco.core.module import Module


class PassThroughModule(Module):
    def forward(self, x, *, cache, **kwargs):
        self.process_data(x, cache=cache, **kwargs)
        return x

    @abstractmethod
    def process_data(self, x, *, cache, **kwargs): ...
