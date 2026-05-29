from collections.abc import Callable
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    overload,
)
from warnings import deprecated

import torch.nn as nn

from saeco.architecture.architecture import ArchitectureBase
from saeco.misc.field_collection import FIELDS, FieldRegistry

if TYPE_CHECKING:
    from saeco.components.losses import Loss


class NonSingular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[False] = False


class Singular(Protocol):
    COLLECTED_FIELD_SINGULAR: Literal[True] = True


class _setup_guarded_cached_property[T](cached_property):  # noqa: N801  # lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """A ``cached_property`` whose body runs only after ``setup()`` finished.

    Accessing it on an architecture instance triggers ``instantiate()`` (which
    runs ``setup()``) if needed, then asserts setup completed. Because
    ``instantiate()`` flips ``_instantiated`` *before* calling ``setup()`` and
    sets ``_setup_complete`` only *after*, this also makes reading such a
    property from *inside* ``setup()`` fail loudly — enforcing the convention
    that ``setup()`` does the stateful configuration of ``self.init`` and the
    cached bodies only read the finished configuration. Cache-once semantics
    (via ``cached_property``) are inherited.
    """

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: Any | None = None) -> T | Self:
        if instance is not None:
            assert isinstance(instance, ArchitectureBase)
            if not instance._instantiated:
                instance.instantiate()
            assert instance._instantiated
            assert instance._setup_complete
        return super().__get__(instance, owner)


class component_prop[T](_setup_guarded_cached_property[T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """A cached, setup-guarded sub-component of an ``Architecture``.

    Like ``@model_prop`` / ``@loss_prop``, accessing it forces ``setup()`` to
    have completed (so the body sees the fully-configured ``self.init``);
    unlike them it is *not* collected as a discoverable field. Use it for
    shared intermediate modules that several models/losses compose and that
    must be built exactly once — e.g. a ``ReuseForward``-wrapped encoder
    referenced by both the magnitude and gate paths of a gated SAE.
    """


class arch_prop[T](  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    _setup_guarded_cached_property[T],
):
    """Base class for the architecture property decorators.

    A setup-guarded ``cached_property`` (see
    :class:`_setup_guarded_cached_property`) that also registers the decorated
    method as a named, discoverable part of an ``Architecture`` (so the
    framework can collect models, losses, etc. by field). Use the concrete
    subclasses ``model_prop`` / ``loss_prop`` / ``aux_model_prop`` rather than
    this directly.

    The discovery bookkeeping lives in the composed ``FieldRegistry``
    (``_registry``); this class is just a holder that forwards to it. The
    setup-guard and cache-once access semantics come from the base.
    """

    COLLECTED_FIELD_SINGULAR = False
    _registry: ClassVar[FieldRegistry] = FIELDS

    def __init__(self, func: Callable[[Any], T]) -> None:
        super().__init__(func)
        self._registry.mark_unowned(self)

    def __set_name__(self, owner: type, name: str) -> None:
        self._registry.claim(
            owner, type(self), name, self, singular=self.COLLECTED_FIELD_SINGULAR
        )
        return super().__set_name__(owner, name)

    @classmethod
    def get_fields(cls, owner: type) -> list[str]:
        return cls._registry.get_fields(owner, cls)

    @overload
    @classmethod
    def get_from_fields(cls: type[NonSingular], inst: object) -> dict[str, T]: ...
    @overload
    @classmethod
    def get_from_fields(cls: type[Singular], inst: object) -> T: ...

    @classmethod
    def get_from_fields(cls, inst: object) -> dict[str, T] | T:
        return cls._registry.get_from_fields(
            inst, cls, singular=cls.COLLECTED_FIELD_SINGULAR
        )


class arch_prop_singular[T](arch_prop[T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    COLLECTED_FIELD_SINGULAR = True

    @classmethod
    def get_from_fields(cls, inst: object) -> T:
        return cls._registry.get_from_fields(inst, cls, singular=True)


class loss_prop[Loss_T: nn.Module](arch_prop[Loss_T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """Declares a training loss on an ``Architecture``.

    Decorate a method that returns a ``Loss``. All ``loss_prop``s are
    collected and optimized during training; their weights come from
    ``train_cfg.coeffs`` keyed by the method name.
    """

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Loss_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> Loss_T | Self:
        return super().__get__(instance, owner)


class metric_prop[Metric_T: nn.Module](arch_prop[Metric_T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Metric_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> Metric_T | Self:
        return super().__get__(instance, owner)


class _model_prop_base[T](arch_prop[T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """
    Base class for model_prop and aux_model_prop.
    Adds on top of arch_prop: methods for attaching losses and metrics to a model.
    - these methods create a new loss_prop or metric_prop where
        the called function/constructor will be called with the
        model as an argument
    """

    @deprecated("define with loss_prop instead for now")
    def add_loss(self, loss: "Loss") -> None:
        raise NotImplementedError

    # # tries to infer whether this is a method (therefore needing self
    # # as the first arg)
    # # or a Loss constructor
    # from saeco.components.losses import Loss

    # assert isinstance(loss, Loss)

    # def get_loss_object(inst):
    #     return loss(self.__get__(inst))

    # return loss_prop(get_loss_object)

    # def add_metric(self, metric):
    #     def _metric(inst):
    #         return metric(self.__get__(inst))

    #     return metric_prop(_metric)


class model_prop[SAE_T: nn.Module](arch_prop_singular[SAE_T], _model_prop_base[SAE_T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """Declares the core model on an ``Architecture``.

    Decorate the single method that builds and returns the main ``SAE``.
    Exactly one ``model_prop`` is expected per architecture; it's what
    gets trained and saved.
    """

    COLLECTED_FIELD_SINGULAR = True

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> SAE_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> SAE_T | Self:
        return super().__get__(instance, owner)

    loss = loss_prop


class aux_model_prop[AuxModel_T: nn.Module](_model_prop_base[AuxModel_T]):  # noqa: N801  # public decorator API; lowercase-class-as-decorator by convention (cf. functools.cached_property)
    """Declares an auxiliary model on an ``Architecture``.

    Decorate a method returning a secondary ``SAE`` (e.g. the gating
    sub-model of a Gated SAE). Aux models run alongside the core model
    and contribute their own ``loss_prop`` losses, but are not the saved
    artifact. Any number may be declared.
    """

    COLLECTED_FIELD_SINGULAR = False

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(
        self, instance: object, owner: type[Any] | None = None
    ) -> AuxModel_T: ...

    def __get__(
        self, instance: object | None, owner: Any | None = None
    ) -> AuxModel_T | Self:
        return super().__get__(instance, owner)


class prop:  # noqa: N801  # namespace of architecture property decorators
    """Discoverable namespace grouping the architecture property decorators.

    Lets you write ``@prop.component`` / ``@prop.model`` / ``@prop.loss`` /
    ``@prop.aux_model`` / ``@prop.metric`` instead of importing each name
    individually. Each attribute is exactly the corresponding standalone
    ``*_prop`` decorator, so the two styles are interchangeable.
    """

    component = component_prop
    model = model_prop
    loss = loss_prop
    aux_model = aux_model_prop
    metric = metric_prop
    arch = arch_prop
