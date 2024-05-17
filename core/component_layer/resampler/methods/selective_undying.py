from jaxtyping import Float
from torch.nn.modules import ReLU
from torch.nn.parameter import Parameter
from sae_components.core.component_layer.component_layer import ComponentLayer
from sae_components.core.component_layer.resampler import (
    ResamplingMethod,
    QueuedResampler,
    ResamplerConfig,
    ResamplerComponent,
)

from sae_components.core import CacheModule, Cache
from sae_components.core.component_layer.resampler.resampler import ResamplingCache
from sae_components.core.nonlinearities.undying import undying_relu
from dataclasses import dataclass
import torch
from sae_components.core.nonlinearities.serializable import SerializableNonlinearity


@dataclass
class SelectiveUndyingResamplerConfig(ResamplerConfig):
    undying_relu: SerializableNonlinearity = SerializableNonlinearity(
        "undying_relu",
        {
            "k": 1,
            "l": 0.01,
            "l_mid_neg": 0.002,
            "l_low_pos": 0.005,
            "l_low_neg": 0.002,
        },
    )
    bias_decay: float = 0.9999
    weight_decay: float = None
    alive_thresh_mul: float = 2
    resample_before_step: bool = True
    wait_to_check_dead: int = 0
    set_dec_to_enc: bool = False
    max_undying: int = None
    add_diff_mae_to_dec: float = None
    add_to_max_acts: float = None
    add_softmax_to_acts: float = None
    softmax_topk: int = None
    grad_to_max_act_decs: float = None
    undying_sq_ema_reset_ratio: float = 0.01
    undying_bias_sq_ema_reset_ratio: float = 0.01
    reset_new_dead: bool = False
    reset_undied: bool = False
    reset_undied_momentum: bool = True
    grad_for_bias: bool = True
    l1_grads_to_undying: float = False


class SelectiveUndyingResampler(ResamplerComponent):
    cfg: SelectiveUndyingResamplerConfig

    def __init__(
        self,
        cfg: SelectiveUndyingResamplerConfig,
        W_next: Parameter | None = None,
        get_optim_fn=None,
    ):
        super().__init__(cfg, W_next, get_optim_fn)
        self.dead = False
        self.nonlinearity = CacheLayerReferredForwardFn(self._nonlinearity)

    def is_check_step(self):
        return (
            self.dead is False
            or super().is_check_step()
            and self.T > self.cfg.wait_to_check_dead
        )

    def is_resample_step(self):
        return True

    @torch.no_grad()
    def resample_callback(self, cache: ResamplingCache, x=None, y_pred=None, y=None):
        inactive_dead = self.dead & (cache.acts == 0).all(dim=0)
        if not self.cfg.grad_for_bias:
            self._layer.cachelayer.b.grad[inactive_dead] = 0
        self._layer.cachelayer.b[inactive_dead] *= self.cfg.bias_decay
        if self.cfg.weight_decay is not None:
            self._layer.cachelayer.W.transpose(-2, -1)[
                inactive_dead
            ] *= self.cfg.weight_decay
        cache.num_undying = ...
        cache.num_undying = self.dead.count_nonzero() if self.dead is not False else 0
        cache.num_dead = ...
        cache.num_dead = self.get_dead_neurons().count_nonzero()
        if self.cfg.set_dec_to_enc:
            self.W_next.transpose(-2, -1)[self.dead] = (
                self._layer.cachelayer.W.transpose(-2, -1)[self.dead]
            )
            self._layer.cachelayer.W.grad.transpose(-2, -1)[
                self.dead
            ] += self.W_next.grad.transpose(-2, -1)[self.dead]

    def check_dead(self):
        if self.dead is False:
            self.dead = self.get_dead_neurons()
        else:
            still_dead = self.dead & self._layer.activations.get_dead_neurons(
                self.cfg.min_viable_count,
                self.cfg.dead_threshold * self.cfg.alive_thresh_mul,
            )
            undied = self.dead & ~still_dead
            new_dead = self.get_dead_neurons()
            if self.cfg.reset_undied:
                self.reset_adam(
                    undied,
                    dead=self.dead | new_dead,
                    reset_momentum=self.cfg.reset_undied_momentum,
                )
            if self.cfg.max_undying is not None:
                new_dead = new_dead & (
                    torch.rand_like(self._layer.activations.freqs)
                    < min(
                        1,
                        (
                            (self.cfg.max_undying - still_dead.count_nonzero())
                            / (
                                self.cfg.max_undying
                                + new_dead.count_nonzero()
                                - still_dead.count_nonzero()
                            )
                        ).item(),
                    )
                )
            if self.cfg.reset_new_dead:
                self.reset_adam(
                    new_dead,
                    dead=self.dead | new_dead,
                    sq_ema_reset_ratio=self.cfg.undying_sq_ema_reset_ratio,
                    bias_sq_ema_reset_ratio=self.cfg.undying_bias_sq_ema_reset_ratio,
                    reset_momentum=False,
                )
            self.dead = new_dead | still_dead
        # self.reset_activation_frequencies(undied)

    def _nonlinearity(self, x, cache):
        if self.dead is False:
            return torch.relu(x)

        relu_only = torch.relu(x)
        out = torch.where(
            self.dead,
            self.cfg.undying_relu(x),
            relu_only,
        )
        cache.acts_spoof = ...
        if isinstance(self.cfg.l1_grads_to_undying, float):
            cache.acts_spoof = out * self.cfg.l1_grads_to_undying + relu_only * (
                1 - self.cfg.l1_grads_to_undying
            )
        elif self.cfg.l1_grads_to_undying:
            cache.acts_spoof = out
        else:
            cache.acts_spoof = relu_only
        if (
            self.cfg.add_to_max_acts
            or self.cfg.add_diff_mae_to_dec
            or self.cfg.grad_to_max_act_decs
            or self.cfg.add_softmax_to_acts
        ):
            i = x[:, self.dead].argmax(dim=0)
        if self.cfg.add_to_max_acts:
            z = torch.zeros_like(out)
            z[:, self.dead][i, torch.arange(len(i))] = self.cfg.add_to_max_acts
            out = out + z
        if self.cfg.add_softmax_to_acts:
            z = torch.zeros_like(out)
            z[:, self.dead] = (
                out[:, self.dead].softmax(dim=0) * self.cfg.add_softmax_to_acts
            )
            if self.cfg.softmax_topk:
                i = z[:, self.dead].topk(self.cfg.softmax_topk, dim=0).indices
                z2 = torch.zeros_like(z)
                z2[:, self.dead][i, torch.arange(len(i))] = z[:, self.dead][i]
            out = out + z
            # cache.nonlinear_argsmaxed = ...
            # cache.nonlinear_argsmaxed = i
        return out

    def _register_parent_layer(self, layer: ComponentLayer):
        super()._register_parent_layer(layer)

        def overwrite_acts(cache: Cache, acts):
            if self.dead is False or not self.cfg.add_to_max_acts:
                return
            return cache.acts_spoof
            # mask = torch.ones_like(acts, dtype=torch.bool)
            # i = cache.nonlinear_argsmaxed
            # mask[:, self.dead][i, torch.arange(len(i))] = 0
            # return acts * mask

        layer.train_cache_template.register_write_callback(
            "acts", overwrite_acts, nice=-10
        )
        assert layer.cachelayer.nonlinearity is self.nonlinearity, (
            layer.cachelayer.W.shape,
            layer.cachelayer.nonlinearity,
            self.nonlinearity,
        )


class CacheLayerReferredForwardFn(CacheModule):
    def __init__(self, forward_fn):
        super().__init__()
        self._referred_forward_fn = forward_fn

    def forward(self, *x, cache):
        return self._referred_forward_fn(*x, cache=cache)
