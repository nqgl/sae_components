from nqgl.mlutils.components.cache import Cache


import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple, Any, Union, Optional, List
import einops
from unpythonic import box
from abc import abstractmethod, ABC


class ActsCache(Cache):
    acts: Float[Tensor, "batch *inst d_out"] = ...


@dataclass
class CacheLayerConfig:
    d_in: int = 0
    d_out: int = 0
    inst: Optional[List[int]] = None


class CacheModule(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, *x, cache: Cache = None, **kwargs):
        raise NotImplementedError


class CacheLayer(CacheModule):
    def __init__(
        self,
        W: Float[Tensor, "*inst d_in d_out"],
        bias: Union[Float[Tensor, "*#inst d_out"], bool],
        b_in=None,
        nonlinearity=torch.nn.ReLU(),
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = W.shape[-2]
        self.d_out = W.shape[-1]
        self.inst = W.shape[:-2]
        # TODO maybe assert all are parameters already?
        self.W = nn.Parameter(W) if isinstance(W, Tensor) else W
        self.ndim_inst = self.W.ndim - 2
        self.b = nn.Parameter(bias) if isinstance(bias, Tensor) else bias
        if b_in is None:
            self.b_pre = 0
        else:
            self.b_pre = nn.Parameter(b_in) if isinstance(b_in, Tensor) else b_in
        self.nonlinearity = nonlinearity
        # self.W = nn.Parameter(W)

    def forward(self, x, cache: Cache, **kwargs):
        cache.x = x
        x = x.view(
            *(x.shape[:1] + (1,) * (self.ndim_inst - (x.ndim - 2)) + x.shape[1:])
        )
        # TODO become fully satisfied w the squeezing
        mul = (x + self.b_pre).unsqueeze(-2) @ self.W
        cache.pre_acts = (pre_acts := mul.squeeze(-2) + self.b)
        cache.acts = (
            acts := (
                self.nonlinearity(pre_acts, cache=cache)
                if isinstance(self.nonlinearity, CacheModule)
                else self.nonlinearity(pre_acts)
            )
        )
        return acts

    @classmethod
    def from_dims(
        cls, d_in, d_out, bias=True, b_in=False, inst=tuple(), cfg=None, **kwargs
    ):
        if cfg is None:
            cfg = CacheLayerConfig(d_in, d_out, inst)
        W = torch.nn.init.kaiming_uniform_(torch.empty(*inst, d_in, d_out))
        b_out = torch.zeros(*inst, d_out) if bias else 0
        b_in = torch.zeros(*inst, d_in) if b_in else None
        return cls(W, b_out, b_in, cfg=cfg, **kwargs)

    @classmethod
    def from_cfg(cls, cfg: CacheLayerConfig):
        return cls.from_dims(cfg.d_in, cfg.d_out, inst=cfg.inst, cfg=cfg)


class CacheProcLayer(CacheModule):
    def __init__(self, cachelayer: CacheModule, train_cache=None, eval_cache=None):
        super().__init__()
        self.cachelayer = cachelayer
        self.train_cache_template: Cache = train_cache or ActsCache()
        self.eval_cache_template: Cache = eval_cache or Cache()
        self.train_process_after_call: set = set()
        self.eval_process_after_call: set = set()

    def forward(self, *x, cache: Cache = None):
        cache = self.prepare_cache(cache)
        acts = self.cachelayer(*x, cache=cache)
        self._update(cache)
        return acts

    def _update(self, cache: Cache):
        if self.training:
            for fn in self.train_process_after_call:
                fn(cache)
        else:
            for fn in self.eval_process_after_call:
                fn(cache)

    def prepare_cache(self, cache: Cache = None):
        b, cache = (cache, None) if isinstance(cache, box) else (None, cache)
        if cache is None:
            cache = self.generate_default_cache()
            if b:
                b << cache
            return cache
        return self.register_to_external_cache(cache)

    def generate_default_cache(self):
        if self.training:
            return self.train_cache_template.clone()
        else:
            return self.eval_cache_template.clone()

    def register_to_external_cache(self, cache: Cache):
        cache += self.generate_default_cache()
        return cache
