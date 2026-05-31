from collections.abc import Iterator, Mapping
from typing import Protocol, cast, overload, runtime_checkable

import attrs
import torch
import torch.nn as nn
from torch import Tensor

from saeco.components.features.optim_reset import (
    FeatureParamType,
    OptimResetValues,
)

IndexType = int | list[int]


class OptimFieldFeatures:
    def __init__(
        self,
        optim: torch.optim.Optimizer,
        fp: "FeaturesView",
        field: str | None = None,
        index=None,
    ):
        self.optim = optim
        self.fp = fp
        self.index = index
        self.field = field

    def parse_index_field(self, i):
        if isinstance(i, tuple) and any(isinstance(el, str) for el in i):
            assert len(i) == 2
            if isinstance(i[0], str):
                field, index = i
            else:
                index, field = i
        elif isinstance(i, str):
            field = i
            index = None
        else:
            index = i
            field = None
        if index is not None and self.index is not None:
            raise KeyError("Index already set")
        if field is not None and self.field is not None:
            raise KeyError("Field already set")

        index = index if index is not None else self.index
        field = field if field is not None else self.field
        assert isinstance(field, str | None)
        return index, field

    @overload
    def __getitem__(self, i: IndexType) -> Mapping[str, Tensor]: ...
    @overload
    def __getitem__(self, i: str) -> Mapping[IndexType, Tensor]: ...
    @overload
    def __getitem__(
        self, i: tuple[str, IndexType] | tuple[IndexType, str]
    ) -> Tensor: ...

    def __getitem__(
        self, i: tuple[str, IndexType] | tuple[IndexType, str] | str | IndexType
    ) -> Tensor | Mapping[str, Tensor] | Mapping[IndexType, Tensor]:
        index, field = self.parse_index_field(i)
        if None in (index, field):
            return cast(
                Mapping[str, Tensor] | Mapping[IndexType, Tensor],
                OptimFieldFeatures(self.optim, self.fp, field, index),
            )
        assert field is not None
        return self.fp.features_transform(self.pstate[field])[index]

    def __setitem__(self, i, v):
        index, field = self.parse_index_field(i)
        if None in (index, field):
            raise KeyError("Index or field not set")
        assert field is not None
        self.fp.features_transform(self.pstate[field])[index] = v

    @property
    def pstate(self) -> Mapping[str, Tensor]:
        return self.optim.state[self.fp.param]

    def keys(self):
        return self.pstate.keys()


def _coerce_optional_fptype(
    v: FeatureParamType | str | None,
) -> FeatureParamType | None:
    return None if v is None else FeatureParamType(v)


@attrs.define(eq=False)
class FeaturesView:
    """A view of an ``nn.Parameter`` along its feature axis, plus the metadata
    the resampler needs to reset that feature.

    ``feature_index`` is the axis of ``param`` that enumerates features — e.g.
    ``0`` for an encoder weight ``[d_dict, d_in]`` or a bias ``[d_dict]``, ``1``
    for a decoder weight ``[d_in, d_dict]``. ``features`` presents ``param``
    transposed so that axis is moved to the front, i.e. axis 0 always indexes
    features; normalization and resampling operate on that view (and write back
    through it, since the transpose shares storage with ``param``).

    ``type`` (a :class:`FeatureParamType`) drives resampling behavior. It is not
    set by callers — :class:`FPInfo` stamps it (via ``attrs.evolve``) from the
    slot the view is placed in. ``field_handlers`` / ``resampler_cfg`` are
    attached later by :meth:`set_cfg`.
    """

    param: nn.Parameter
    feature_index: int
    type: FeatureParamType | None = attrs.field(
        default=None, converter=_coerce_optional_fptype
    )
    resampled: bool = True
    reset_optim_on_resample: bool = True
    field_handlers: OptimResetValues | None = None
    resampler_cfg: object = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, FeaturesView):
            return False
        return (
            self.param is other.param
            and self.feature_index == other.feature_index
            and self.field_handlers == other.field_handlers
            and self.resampled == other.resampled
            and self.type == other.type
            and self.resampler_cfg == other.resampler_cfg
        )

    def __hash__(self):
        return hash(
            (
                self.param,
                self.feature_index,
                self.field_handlers,
                self.resampled,
                self.type,
            )
        )

    def set_cfg(self, cfg):
        if self.resampler_cfg is cfg:
            return
        if self.resampler_cfg is not None:
            raise ValueError("Cannot set cfg twice")
        self.field_handlers = OptimResetValues(cfg.optim_reset_cfg)
        self.resampler_cfg = cfg

    def features_transform(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(0, self.feature_index)

    def reverse_transform(self, tensor: Tensor) -> Tensor:
        return self.features_transform(tensor)

    @property
    def features(self) -> Tensor:  # TODO is there a name with more clarity?
        # is "data" right? thinking that might be kinda wrong
        return self.features_transform(self.param)

    @property
    def grad(self) -> Tensor | None:
        if self.param.grad is None:
            return None
        return self.features_transform(self.param.grad)

    @classmethod
    def get_optim(cls, optim):
        try:
            from torchlars import LARS

            if isinstance(optim, LARS):
                optim = optim.optim
        except ImportError:
            pass
        return optim

    def optimstate(self, optim) -> OptimFieldFeatures:
        try:
            from torchlars import LARS

            if isinstance(optim, LARS):
                optim = optim.optim
        except ImportError:
            pass
        return OptimFieldFeatures(optim, self)

    @torch.no_grad()
    def resample(
        self, *, indices, new_directions, bias_reset_value, optim: torch.optim.Optimizer
    ):
        if isinstance(new_directions, tuple):
            assert len(new_directions) == 2
            if self.type == "enc":
                new_directions = new_directions[0]
            elif self.type == "dec":
                new_directions = new_directions[1]
            elif self.type == "other":
                new_directions = new_directions[0]
        if not self.type == "bias":
            new_directions = new_directions / new_directions.norm(dim=1, keepdim=True)
        if self.type == "enc":
            avg_other_norm = 1
            if not indices.all():
                avg_other_norm = self.features[~indices].norm(dim=1).mean()
            new_directions *= avg_other_norm * 0.2

        if self.type == FeatureParamType.bias:
            self.features[indices] = bias_reset_value
        else:
            self.features[indices] = new_directions
        if not self.reset_optim_on_resample:
            return
        optim_state = self.optimstate(optim)
        fields = set(optim_state.keys()) - set(self.field_handlers.skips)

        for field in fields:
            if self.get_optim(optim).state[field] == {}:
                continue
            field_state = optim_state[field]
            assert field_state[:].shape == self.features.shape, (
                f"{field}: {field_state[:].shape} != {self.features.shape}"
            )

            optim_state[field, indices] = self.field_handlers.get_value(
                field=field,
                param=self,
                ft_optim_field_state=optim_state[field],
                feat_mask=indices,
                new_directions=new_directions,
            )


@attrs.define
class FPInfo:
    """The feature parameters a :class:`HasFeatures` module exposes, grouped by
    :class:`FeatureParamType`.

    One optional slot per type. The slot a :class:`FeaturesView` lands in *is*
    its type: on construction each present view is replaced by a typed copy
    (``attrs.evolve``) — the caller's object is never mutated — so call sites no
    longer pass a type at all; a view that already carries a conflicting type is
    a loud error. A module that genuinely needs two params of one type cannot
    express it here — that is intentional (fail loud), revisit the container
    then rather than papering over it.
    """

    enc: FeaturesView | None = None
    dec: FeaturesView | None = None
    bias: FeaturesView | None = None
    other: FeaturesView | None = None

    def __attrs_post_init__(self):
        # The slot a view lands in *is* its type. Rather than mutate the
        # caller's object, replace each slot with a typed copy (attrs.evolve);
        # a view that already carries a conflicting type is a loud error.
        for t in FeatureParamType:
            fp = getattr(self, t.value)
            if fp is None:
                continue
            if fp.type is None:
                setattr(self, t.value, attrs.evolve(fp, type=t))
            elif fp.type != t:
                raise ValueError(
                    f"FeaturesView placed in the {t.value!r} slot already carries "
                    f"a conflicting type {fp.type!r}"
                )

    def __iter__(self) -> Iterator[FeaturesView]:
        for t in FeatureParamType:
            fp = getattr(self, t.value)
            if fp is not None:
                yield fp

    def get(self, which: FeatureParamType) -> FeaturesView:
        which = FeatureParamType(which)
        fp = getattr(self, which.value)
        if fp is None:
            raise KeyError(
                f"No {which.value!r} feature parameter present; "
                f"have {[t.value for t in FeatureParamType if getattr(self, t.value)]}"
            )
        return fp


# Keep the container's slots and the type enum in lockstep: adding a
# FeatureParamType member without a matching slot (or vice versa) fails loudly
# here at import rather than silently dropping that type from iteration.
assert {f.name for f in attrs.fields(FPInfo)} == {t.value for t in FeatureParamType}, (
    "FPInfo slots must match FeatureParamType members exactly"
)


def get_featuresparams(model: nn.Module) -> set[FeaturesView]:
    l: set[FeaturesView] = set()
    for m in model.modules():
        if isinstance(m, HasFeatures):
            l |= set(m.features)
    d = {}
    for fp in l:
        if fp.param in d:
            other = d[fp.param]
            assert other == fp, f"{other} != {fp}"
            raise ValueError(
                f"Duplicate feature parameter {fp}. implement __eq__ and change this "
                "check to just (intelligently) deduplicate and check for inconsistency"
            )
        d[fp.param] = fp
    return l


def get_resampled_params(model: nn.Module):
    for fp in get_featuresparams(model):
        if fp.resampled:
            yield fp


@runtime_checkable
class HasFeatures(Protocol):
    @property
    def features(self) -> FPInfo: ...
