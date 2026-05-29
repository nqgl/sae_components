"""Behavioural tests for the architecture property decorators.

These go through the *real* ``arch_prop`` family (and a ``typeacc_method``
hook) rather than the registry directly, to lock the holder wiring after the
refactor that moved the shared bookkeeping into a composed ``FieldRegistry``:
registration/enumeration, singular enforcement, bucket separation, the
cached-value ``__get__`` gating, and that arch props and training hooks share
one registry without colliding.
"""

import pytest

from saeco.architecture import ArchitectureBase
from saeco.architecture.arch_prop import (
    aux_model_prop,
    loss_prop,
    model_prop,
)
from saeco.components.type_acc_methods import (
    PostStepHook,
    post_step_hook,
    typeacc_method,
)


def test_props_register_into_separate_buckets_in_order():
    class Arch:
        @model_prop
        def model(self): ...

        @loss_prop
        def l1(self): ...

        @loss_prop
        def l2(self): ...

        @aux_model_prop
        def aux(self): ...

    assert model_prop.get_fields(Arch) == ["model"]
    assert loss_prop.get_fields(Arch) == ["l1", "l2"]
    assert aux_model_prop.get_fields(Arch) == ["aux"]


def test_model_prop_is_singular():
    with pytest.raises(AttributeError, match="singular"):

        class TwoModels:
            @model_prop
            def m1(self): ...

            @model_prop
            def m2(self): ...


def test_loss_and_aux_props_allow_multiple():
    class Arch:
        @loss_prop
        def la(self): ...

        @loss_prop
        def lb(self): ...

        @aux_model_prop
        def ax(self): ...

        @aux_model_prop
        def ay(self): ...

    assert loss_prop.get_fields(Arch) == ["la", "lb"]
    assert aux_model_prop.get_fields(Arch) == ["ax", "ay"]


class _FakeArch(ArchitectureBase):
    """Minimal ``ArchitectureBase`` subclass — the type ``arch_prop.__get__``
    now gates on — that skips the heavy ``RunConfig`` / ``Initializer`` setup
    and just flips the instantiate/setup flags so cached-value access works."""

    def __init__(self):
        self._instantiated = False
        self._setup_complete = True

    def instantiate(self, inst_cfg=None):
        self._instantiated = True

    def setup(self): ...

    def run_training(self): ...

    @model_prop
    def the_model(self):
        return "MODEL"

    @loss_prop
    def loss_a(self):
        return "LA"

    @loss_prop
    def loss_b(self):
        return "LB"


def test_get_triggers_instantiate_and_caches():
    inst = _FakeArch()
    assert inst._instantiated is False
    assert inst.the_model == "MODEL"
    # __get__ ran instantiate() on first access...
    assert inst._instantiated is True
    # ...and the cached_property semantics mean the same object is reused
    assert inst.the_model is inst.the_model


def test_get_from_fields_singular_returns_the_model():
    inst = _FakeArch()
    assert model_prop.get_from_fields(inst) == "MODEL"


def test_get_from_fields_nonsingular_returns_named_dict():
    inst = _FakeArch()
    assert loss_prop.get_from_fields(inst) == {"loss_a": "LA", "loss_b": "LB"}


def test_inherited_props_visible_on_subclass():
    class Base:
        @loss_prop
        def base_loss(self): ...

    class Sub(Base): ...

    assert loss_prop.get_fields(Sub) == ["base_loss"]


def test_subclass_props_merge_with_base():
    class Base:
        @loss_prop
        def base_loss(self): ...

    class Sub(Base):
        @loss_prop
        def sub_loss(self): ...

    # a subclass that adds its own loss sees the base's followed by its own,
    # rather than shadowing the base's
    assert loss_prop.get_fields(Sub) == ["base_loss", "sub_loss"]
    assert loss_prop.get_fields(Base) == ["base_loss"]


def test_arch_props_and_hooks_share_one_registry_without_colliding():
    # the refactor unified both families onto a single composed registry
    assert arch_prop_registry() is typeacc_method._registry

    class Arch:
        @loss_prop
        def la(self): ...

    class Hooked:
        @post_step_hook
        def step(self): ...

    # each family sees only its own bucket; no bleed across families/owners
    assert loss_prop.get_fields(Arch) == ["la"]
    assert PostStepHook.get_fields(Hooked) == ["step"]
    assert PostStepHook.get_fields(Arch) == []
    assert loss_prop.get_fields(Hooked) == []


def arch_prop_registry():
    return loss_prop._registry
