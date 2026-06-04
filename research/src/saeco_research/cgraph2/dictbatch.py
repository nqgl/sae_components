"""Typed batch contracts for multi-input graphs -- the typed surface over
``field()`` (no string accesses).

Declare a batch as a class of annotated fields; the class-level field is a typed
``FieldRef`` (so ``Batch.input`` reads as ``torch.Tensor`` to the type checker and
to you), while a *constructed* ``Batch(input=..., ...)`` exposes the values as
instance attributes::

    class Batch(DictBatch):
        input: torch.Tensor
        target: torch.Tensor
        weight: float

    class TranscoderLoss(GraphSpec):
        x = field(Batch.input)     # typed handle reading the seeded batch's .input
        y = field(Batch.target)    # distinct from x -- the transcoder case (y != x)
        w = field(Batch.weight)
        recon = op(torch.tanh)(x)
        output = w * op(_l2)(recon, y)

    g = TranscoderLoss()
    g.forward(Batch(input=x, target=y, weight=2.0), cache=...)

The trick is that ``Batch.input`` is annotated ``torch.Tensor`` but the metaclass
*also* assigns it a ``FieldRef`` at class creation -- pyright reports the
annotation type (``Tensor``) for the class access while the runtime value carries
the field name. ``field(Batch.input)`` reads the name off it (see
``cgraph2.core.field``, which duck-types ``str | FieldRef``).

To not be *forced* onto a batch input, declare the field as an overridable slot
whose default reads the batch -- ``x: Arg = arg(default=field(Batch.input))`` --
and bind ``x`` to a plain value / ``Input`` at construction (Glen's "set y and w,
pass z=Input" case). Field-as-handle (above) is the common path; field-as-slot is
the escape hatch, and both are just the existing ``field`` / ``arg`` machinery.
"""

from __future__ import annotations

from typing import Any, ClassVar, dataclass_transform


class FieldRef:
    """Class-level handle for a :class:`DictBatch` field: typed (to the checker) as
    the field's value type, but at runtime a carrier of ``(owner, name, type)``.
    ``field(Batch.input)`` reads ``.name`` off it."""

    def __init__(self, owner: str, name: str, typ: Any) -> None:
        self.owner = owner
        self.name = name
        self.type = typ

    def __repr__(self) -> str:
        return f"{self.owner}.{self.name}"


@dataclass_transform(eq_default=False)
class _BatchMeta(type):
    def __new__(mcs, name, bases, ns):
        cls = super().__new__(mcs, name, bases, ns)
        types: dict[str, Any] = {}
        for base in bases:
            types.update(getattr(base, "_field_types", {}))
        types.update(
            {k: t for k, t in ns.get("__annotations__", {}).items() if k[:1] != "_"}
        )
        # the class-level field is the typed FieldRef; an *instance*'s attribute
        # (set in __init__) shadows it with the actual value.
        for k, t in types.items():
            setattr(cls, k, FieldRef(name, k, t))
        cls._field_types = types  # type: ignore[attr-defined]
        return cls


class DictBatch(metaclass=_BatchMeta):
    """A typed batch contract (see module docstring). Permissive by default: a
    constructed batch carries its declared fields *and* any extra ones passed
    (the "auto other fields" behavior -- e.g. runtime-added per-loss weights)."""

    _field_types: ClassVar[dict[str, Any]] = {}

    def __init__(self, **values: Any) -> None:
        for key, value in values.items():  # instance attrs shadow class FieldRefs
            setattr(self, key, value)

    def __getattribute__(self, name: str) -> Any:
        # instance access of a declared-but-unprovided field would otherwise fall
        # back to the class-level FieldRef (a silent wrong value); make it loud.
        value = super().__getattribute__(name)
        if type(value) is FieldRef:
            raise AttributeError(
                f"field {name!r} was not provided to this {type(self).__name__}"
            )
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        declared = ", ".join(
            f"{k}={'set' if k in self.__dict__ else 'MISSING'}"
            for k in type(self)._field_types
        )
        extra = [k for k in self.__dict__ if k not in type(self)._field_types]
        tail = f" +{extra}" if extra else ""
        return f"{type(self).__name__}({declared}){tail}"

    def missing_fields(self) -> list[str]:
        """Declared fields not provided -- a graph reading them will fail loud."""
        return [k for k in type(self)._field_types if k not in self.__dict__]
