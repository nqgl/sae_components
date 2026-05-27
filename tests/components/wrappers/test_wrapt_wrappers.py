"""Tests for wrapt-based module wrappers and wrapper-based training hooks."""

import torch
import torch.nn as nn

from saeco.components.features.linear_type import LinDecoder, LinDecoderMixin
from saeco.components.hooks.clipgrad import ClipGrad
from saeco.components.hooks.feature_hooks import (
    NormFeatures,
    OrthogonalizeFeatureGrads,
)
from saeco.components.wrap import WrapsModule
from saeco.core.module import Module
from saeco.core.reused_forward import ReuseForward
from saeco.initializer.linear_factory import LinearFactory
from saeco.trainer.call_training_hooks import do_post_backward, do_post_step


class CountingPostStep(WrapsModule):
    def __init__(self, module):
        super().__init__(module)
        self._self_count = 0

    @property
    def count(self):
        return self._self_count

    def post_step_hook(self):
        self._self_count += 1


class PassiveWrapper(WrapsModule):
    pass


def test_wraps_module_is_transparent_to_torch_module_apis():
    lin = nn.Linear(3, 2)
    wrapped = PassiveWrapper(lin)
    model = nn.Sequential(wrapped)

    assert isinstance(wrapped, nn.Linear)
    assert isinstance(wrapped, nn.Module)
    assert not isinstance(wrapped, Module)
    assert wrapped.weight is lin.weight
    assert list(model.state_dict()) == ["0.weight", "0.bias"]
    assert model(torch.randn(4, 3)).shape == (4, 2)


def test_reuse_forward_uses_wraps_module_transparency():
    lin = nn.Linear(3, 2)
    wrapped = ReuseForward(lin)
    model = nn.Sequential(wrapped)

    assert isinstance(wrapped, nn.Linear)
    assert isinstance(wrapped, Module)
    assert wrapped.weight is lin.weight
    assert list(model.state_dict()) == ["0.weight", "0.bias"]


def test_nested_wrappers_visit_each_hook_once():
    inner = CountingPostStep(nn.Linear(3, 2))
    outer = PassiveWrapper(inner)
    model = nn.Sequential(outer)

    do_post_step(model)

    assert inner.count == 1


def test_reused_nested_wrappers_do_not_duplicate_delegated_plain_hooks():
    wrapped = PassiveWrapper(CountingPostStep(nn.Linear(3, 2)))
    model = nn.Sequential(wrapped, wrapped)

    do_post_step(model)

    assert wrapped.__wrapped__.count == 1
    rewrapped = CountingPostStep(wrapped)
    do_post_step(nn.Sequential(rewrapped, wrapped, model))
    assert rewrapped.count == 1
    assert rewrapped.__wrapped__.count == 2


def test_wrapped_class_registers_as_module():
    class TorchModule(nn.Module):
        def __init__(self, item):
            super().__init__()
            self.item = item
            self.lin = nn.Linear(3, 2)

    linear = nn.Linear(3, 2)
    wrapped = PassiveWrapper(linear)
    wrapped2 = PassiveWrapper(wrapped)
    m0 = TorchModule(linear)
    m1 = TorchModule(wrapped)
    m2 = TorchModule(wrapped2)

    assert isinstance(wrapped, nn.Module)
    assert isinstance(wrapped2, nn.Module)
    assert isinstance(PassiveWrapper(wrapped2), nn.Module)
    assert len(list(m0.modules())) == len(list(m1.modules())) == len(list(m2.modules()))
    assert (
        len(list(m0.parameters()))
        == len(list(m1.parameters()))
        == len(list(m2.parameters()))
    )


def test_norm_features_wrapper_normalizes_decoder_features():
    lin = LinDecoder(3, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[3.0, 0.0, 4.0], [0.0, 8.0, 0.0]]))
    model = nn.Sequential(NormFeatures(lin))

    do_post_step(model)

    norms = lin.features["weight"].features.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_orthogonalize_feature_grads_wrapper_removes_radial_grad_component():
    lin = LinDecoder(3, 2, bias=False)
    model = nn.Sequential(OrthogonalizeFeatureGrads(lin))

    model(torch.randn(5, 3)).sum().backward()
    do_post_backward(model)

    fp = lin.features["weight"]
    dec_normed = fp.features / fp.features.norm(dim=-1, keepdim=True)
    radial_grad = (fp.grad * dec_normed).sum(dim=-1)
    assert torch.allclose(radial_grad, torch.zeros_like(radial_grad), atol=1e-5)


def test_factory_add_wrapper_keeps_state_dict_clean():
    factory = LinearFactory(4, 3, mixins=[LinDecoderMixin])
    factory.add_wrapper(NormFeatures)
    decoder = factory.get()

    assert isinstance(decoder, nn.Linear)
    assert isinstance(decoder, Module) is False
    assert list(decoder.state_dict()) == ["weight", "bias"]

    parent = nn.Module()
    parent.decoder = decoder

    assert list(parent.state_dict()) == ["decoder.weight", "decoder.bias"]


def test_factory_add_wrapper_partial_binding():
    factory = LinearFactory(4, 3, mixins=[LinDecoderMixin])
    factory.add_wrapper(NormFeatures, max_only=True)
    decoder = factory.get()

    with torch.no_grad():
        decoder.weight.mul_(0.1)
    pre = decoder.features["weight"].features.norm(dim=-1).clone()

    do_post_step(decoder)

    post = decoder.features["weight"].features.norm(dim=-1)
    assert torch.allclose(pre, post)


def test_linear_factory_add_wrapper_binds_wrapper_parameters():
    factory = LinearFactory(3, 2, mixins=[LinDecoderMixin])
    factory.add_wrapper(NormFeatures, max_only=True)

    decoder = factory.get()

    assert decoder._self_max_only is True


def test_clipgrad_wrapper_clips_norm():
    lin = nn.Linear(3, 2)
    wrapped = ClipGrad(lin, max_norm=0.5)
    nn.Sequential(wrapped)(torch.randn(4, 3)).sum().backward()

    pre_norm = torch.cat([p.grad.flatten() for p in lin.parameters()]).norm()
    assert pre_norm > 0.5
    do_post_backward(nn.Sequential(wrapped))
    post_norm = torch.cat([p.grad.flatten() for p in lin.parameters()]).norm()
    assert post_norm <= 0.5 + 1e-5
