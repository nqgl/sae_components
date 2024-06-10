from saeco.core.component_layer.layercomponent import LayerComponent
from saeco.core.config import WandbDynamicConfig
from saeco.core.cache import Cache, CacheSpec

from saeco.core.component_layer import ComponentLayer
import torch.nn as nn
from dataclasses import dataclass
import torch
from typing import Optional
from abc import abstractmethod
from torch import Tensor
from jaxtyping import Bool, Float
import torch.nn.functional as F
from saeco.core.component_layer.resampler.adam_resetter import AdamResetter


@dataclass
class ResamplerConfig(WandbDynamicConfig):
    num_to_resample: int = 128
    dead_threshold: float = 3e-6
    min_viable_count: int = 10_000
    check_frequency: int = 100
    norm_encoder_proportional_to_alive: bool = True
    reset_all_freqs_interval: int = 10000
    reset_all_freqs_offset: int = 0
    normalized_encoder_multiplier: float = 0.2
    negative_bias_multiplier: float = 6
    sq_ema_reset_ratio: float = 1
    bias_sq_ema_reset_ratio: float = None

    reset_adam: bool = True


class ResamplingCache(CacheSpec):
    # dead_neurons: torch.Tensor
    saved_x: torch.Tensor
    # resample: Callable
    # saved_y_pred: torch.Tensor


class ResamplerComponent(LayerComponent):
    _default_component_name = "resampler"
    _requires_component = ["activations", "resampling_method"]
    train_cache_watch = []
    eval_cache_watch = []
    resampling_method: "ResamplingMethod"

    def __init__(
        self,
        cfg: ResamplerConfig,
        W_next: Optional[nn.Parameter] = None,
        get_optim_fn=None,
    ):
        self.cfg = cfg
        self.W_next = W_next
        self.T = 0
        self._layer: ComponentLayer = None
        self.get_optim_fn = get_optim_fn

    @property
    def optim(self):
        return self.get_optim_fn()

    def _register_parent_layer(self, layer: ComponentLayer):
        super()._register_parent_layer(layer)
        self._layer.train_cache_template.register_write_callback(
            "x", self._resample_train_hook_x
        )

    def _resample_train_hook_x(self, cache: ResamplingCache, x):
        self._step()
        if self.is_resample_step():
            # cache.resample = ...
            # cache.resample = lambda x, y_pred, y: self.resample_callback(
            #     cache, x, y_pred, y
            # )
            cache._parent.x = ...
            cache._parent.y = ...
            cache._parent.y_pred = ...
            cache.resample = ...
            cache.resample = lambda **kwargs: self.resample_callback(cache, **kwargs)

    # in the future, can denote this with something like:
    # @ComponentLayer.hook.cache.train.x
    # -> set a class field of any LayerComponent that holds hooks to register
    # and then ComponentLayer is responsible for registering/calling them appropriately
    def is_resample_step(self):
        return self.is_check_step()

    def is_check_step(self):
        return self.T % self.cfg.check_frequency == 0 and self._layer.training

    def _step(self):
        if self._layer.training:
            self.T += 1

    @property
    def freqs(self):
        return self._layer.activations.freqs

    def get_dead_neurons(self):
        return self._layer.activations.get_dead_neurons(
            self.cfg.min_viable_count, self.cfg.dead_threshold
        )

    def reset_activation_frequencies(self, mask=None):
        self._layer.activations.reset_freqs(mask)

    @torch.no_grad()
    def reset_neurons(
        self,
        new_directions: Float[Tensor, "nnz d_in"],
        to_reset: Tensor,  # Int[Tensor] or Boool[Tensor]?
        W_next_directions: Optional[Tensor] = None,
        b_reset_values=None,
    ):
        if isinstance(to_reset, tuple):
            self.reset_neurons_from_index(
                new_directions,
                to_reset,
                W_next_directions=W_next_directions,
                b_reset_values=b_reset_values,
            )
        elif to_reset.dtype == torch.bool:
            self.reset_neurons_from_mask(
                new_directions,
                to_reset,
                W_next_directions=W_next_directions,
                b_reset_values=b_reset_values,
            )
        else:
            self.reset_neurons_from_index(
                new_directions,
                to_reset,
                W_next_directions=W_next_directions,
                b_reset_values=b_reset_values,
            )

    @torch.no_grad()
    def reset_neurons_from_mask(
        self,
        new_directions: Float[Tensor, "nnz d_in"],
        to_reset: Bool[Tensor, "*#inst d_out"],
        W_next_directions: Optional[Tensor] = None,
        b_reset_values=None,
    ):  # this may be wrong
        assert False, "not implemented"
        assert new_directions.shape[0] == torch.count_nonzero(to_reset)
        if self.W_next is not None:
            self.W_next.data[to_reset] = self.proc_W_next_directions(new_directions)
        self._layer.cachelayer.W.data.transpose(-2, -1)[to_reset] = (
            self.proc_W_directions(new_directions)
        )
        self._layer.cachelayer.b.data[to_reset] = self.proc_bias_directions(
            new_directions
        )

    @torch.no_grad()
    def reset_neurons_from_index(
        self,
        new_directions: Float[Tensor, "nnz d_out"],
        to_reset: Bool[Tensor, "nnz d_out"],
        W_next_directions: Optional[Tensor] = None,
        b_reset_values=None,
    ):
        assert (
            isinstance(to_reset, tuple)
            and len(to_reset) == self._layer.cachelayer.W.ndim - 1
        ) or (
            to_reset.ndim == 2
            and to_reset.shape[1] == self._layer.cachelayer.W.ndim - 1
        )

        if to_reset[0].shape[0] == 0:
            return
        if self.W_next is not None:
            self.W_next.data.transpose(-2, -1)[to_reset] = self.proc_W_next_directions(
                new_directions if W_next_directions is None else W_next_directions
            )
        W_dirs = self.proc_W_directions(new_directions)
        self._layer.cachelayer.W.data.transpose(-2, -1)[to_reset] = W_dirs
        self._layer.cachelayer.b.data[to_reset] = self.proc_bias_directions(W_dirs)
        if self.cfg.reset_adam:
            self.reset_adam(to_reset=to_reset)

    def reset_adam(
        self,
        to_reset,
        dead=None,
        sq_ema_reset_ratio=None,
        bias_sq_ema_reset_ratio=None,
        reset_momentum=True,
    ):
        dead = dead if dead is not None else self.get_dead_neurons()
        cl_resetter = AdamResetter(
            self._layer.cachelayer,
            sq_ema_reset_ratio=sq_ema_reset_ratio or self.cfg.sq_ema_reset_ratio,
        )
        cl_resetter.W.transpose(-2, -1)[to_reset](
            self.optim, alive_indices=~dead, reset_momentum=reset_momentum
        )
        cl_resetter.b[to_reset](
            self.optim,
            alive_indices=~dead,
            sq_ema_reset_ratio=bias_sq_ema_reset_ratio
            or self.cfg.bias_sq_ema_reset_ratio
            or sq_ema_reset_ratio
            or self.cfg.sq_ema_reset_ratio,
            reset_momentum=reset_momentum,
        )

        if self.W_next is not None:
            self_resetter = AdamResetter(
                self,
                sq_ema_reset_ratio=sq_ema_reset_ratio or self.cfg.sq_ema_reset_ratio,
            )
            self_resetter.W_next.transpose(-2, -1)[to_reset](
                self.optim, alive_indices=~dead, reset_momentum=reset_momentum
            )
            # self_resetter.W_next[to_reset](self.optim, alive_indices=~dead)

    # def proc_directions(self, W_dirs: Tensor, W_next_dirs: Optional[Tensor] = None, bias_values: Optional[Tensor] = None):

    def proc_W_next_directions(self, new_directions: Float[Tensor, "nnz d_out"]):
        return F.normalize(new_directions, dim=-1)

    def proc_W_directions(self, new_directions: Float[Tensor, "nnz d_in"]):
        dead = self.get_dead_neurons_for_norm()
        if torch.all(dead):
            print("warning: all neurons dead")
            return F.normalize(new_directions, dim=-1)
        alives = self._layer.cachelayer.W.transpose(-2, -1)[~dead]
        return (
            F.normalize(new_directions, dim=-1)
            * alives.norm(dim=-1).mean()
            * self.cfg.normalized_encoder_multiplier
        )

    def proc_bias_directions(self, new_directions: Float[Tensor, "nnz d_out"]):
        b = (
            -1
            * self.cfg.negative_bias_multiplier
            * new_directions[0].pow(2).sum().pow(0.5)
        )
        print("b", b)
        print(new_directions[0].pow(2).sum().pow(0.5))
        if len(new_directions) > 3:
            print(new_directions[2].norm())
            print(new_directions[3].norm())
        print(new_directions.shape)
        print("--")
        return b

    def get_dead_neurons_for_norm(self):
        return self.get_dead_neurons()

    @abstractmethod
    def resample_callback(self, cache, x=None, y_pred=None, y=None): ...

    def _update_from_cache(self, cache: ResamplingCache, **kwargs):
        # training = kwargs.get("training", False)
        # if training:
        #     self._step()
        if (
            self.T - self.cfg.reset_all_freqs_offset
        ) % self.cfg.reset_all_freqs_interval == 0:
            self.reset_activation_frequencies()

        if self.is_check_step():
            self.check_dead()
        # if self.is_resample_step():

    def get_neurons_to_resample(self):
        return self.get_dead_neurons()

    @abstractmethod
    def check_dead(self):
        pass


class ResamplingMethod(ResamplerComponent):
    # _default_component_name = "resampling_method"
    # _requires_component = ["resampler"]

    def __init__(
        self, cfg: ResamplerConfig, W_next: Optional[nn.Parameter] = None, **kwargs
    ):
        super().__init__(cfg=cfg, W_next=W_next, **kwargs)
        self.cumulative_num_resampled = 0

    # @abstractmethod
    @torch.inference_mode()
    def resample_callback(self, cache: ResamplingCache, x=None, y_pred=None, y=None):
        x = x if x is not None else cache._ancestor.search("x")[0].x
        y = y if y is not None else cache._ancestor.search("y")[0].y
        y_pred = (
            y_pred if y_pred is not None else cache._ancestor.search("y_pred")[0].y_pred
        )
        dead = self.get_neurons_to_resample()
        directions = self._get_directions(cache, x, y_pred, y)
        if isinstance(directions, tuple):
            if len(directions) == 2:
                directions, W_next_directions = directions
                b_reset_values = None
            else:
                directions, W_next_directions, b_reset_values = directions
        else:
            W_next_directions = None
            b_reset_values = None
        if dead is None or directions is None:
            cache.num_resampled = 0
            return
        if dead.dtype == torch.bool:
            to_reset = dead.nonzero()
        else:
            to_reset = dead
        to_reset = to_reset[: min(directions.shape[0], self.cfg.num_to_resample)]
        W_next_directions = (
            W_next_directions[: to_reset.shape[0]]
            if W_next_directions is not None
            else None
        )
        b_reset_values = (
            b_reset_values[: to_reset.shape[0]] if b_reset_values is not None else None
        )

        directions = directions[: to_reset.shape[0]]
        cache.num_resampled = to_reset.shape[0]
        self.cumulative_num_resampled += to_reset.shape[0]
        cache.cumulative_num_resampled = self.cumulative_num_resampled
        to_reset = to_reset.unbind(-1)
        self.reset_neurons(directions, to_reset, W_next_directions=W_next_directions)

    def reset_neurons(
        self,
        new_directions: Tensor,
        to_reset: Tensor,
        W_next_directions: Optional[Tensor] = None,
        b_reset_values=None,
    ):
        super().reset_neurons(
            new_directions, to_reset, W_next_directions=W_next_directions
        )
        mask = self.get_dead_neurons()
        mask[:] = False
        mask[to_reset] = True
        self.reset_activation_frequencies(mask)

    def _get_directions(self, cache: ResamplingCache, x, y_pred, y):
        return self.get_directions(cache, x, y_pred, y)

    @abstractmethod
    def get_directions(self, cache, x, y_pred, y): ...

    # def resample_from_new_directions(self, ):


class GeneratedBatchResampler(ResamplingMethod):
    def __init__(
        self,
        cfg: ResamplerConfig,
        buffer,
        forward_model,
        W_next: Optional[nn.Parameter] = None,
        **kwargs,
    ):
        super().__init__(cfg=cfg, W_next=W_next, **kwargs)
        self.buffer = buffer
        self.forward_model = forward_model

    def _get_directions(self, cache, x, y_pred, y):
        y_l = []
        y_pred_l = []
        x_l = []
        with torch.inference_mode():
            for i in range(...):
                x = self.buffer.next()
                if isinstance(x, tuple):
                    x, y = x
                else:
                    y = x
                y_pred_l.append(self.forward_model(x))
                y_l.append(y)
                x_l.append(x)
        y = torch.cat(y_l)
        y_pred = torch.cat(y_pred_l)
        x = torch.cat(x_l)
        return self.get_directions(cache, x, y_pred, y)
        ...


class RandomResamplingDirections(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        return torch.randn(self.cfg.num_to_resample, x.shape[-1], device=x.device)


class DiffResamplingDirections(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        return y - y_pred


class DiffDecYEncResamplingDirections(ResamplingMethod):

    def _get_directions(self, cache: ResamplingCache, x, y_pred, y):
        return super()._get_directions(cache, cache.x, y_pred, y)

    def get_directions(self, cache: Cache, x, y_pred, y):
        return x, y - y_pred


class YResamplingDirections(ResamplingMethod):

    def get_directions(self, cache, x, y_pred, y):
        return y


class SVDResampling(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        print("x", x.shape)
        print("y_pred", y_pred.shape)
        print("y", y.shape)
        u, s, v = torch.svd(y - y_pred)
        print("u", u.shape)
        print("s", s.shape)
        print("v", v.shape)
        # k = (self.cfg.num_to_resample + 1) // 2
        k = self.cfg.num_to_resample
        sort = torch.argsort(s.abs(), descending=True)
        # best_k_directions = v[sort[:k]] * s[sort[:k]].unsqueeze(-1)
        print("v", v.shape)
        best_k_directions = v[sort[:k]] * s[sort[:k]].unsqueeze(-1)
        print("bkd", best_k_directions.shape)
        # perf = (best_k_directions @ (y - y_pred).transpose(-2, -1)).mean(
        #     dim=-1
        # )  # (k batch).mean(batch)
        perf = ((y - y_pred) @ best_k_directions.transpose(-2, -1)).pow(3).mean(dim=0)
        print("perf", perf.shape)
        dirs = best_k_directions * torch.sign(perf).unsqueeze(-1)
        return dirs

        return torch.cat((best_k_directions, -best_k_directions), dim=0)
        signs = torch.sign(perf)
        return dirs * signs.unsqueeze(-1)
