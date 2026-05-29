"""Unit tests for the composed field-collection backbone.

These exercise ``FieldRegistry`` directly with *fresh* instances and toy
owner/descriptor-type stand-ins, so they never touch the process-global
``FIELDS`` registry that the real decorators share. Behaviour here is what
``arch_prop`` and ``typeacc_method`` both forward to.
"""

import pytest

from saeco.misc.field_collection import FieldRegistry


class Cat:
    """Stand-in for a descriptor *type* (the per-bucket key)."""


class OtherCat:
    """A second, unrelated descriptor type."""


class _Descriptor:
    """A minimal stand-in descriptor: hashable by identity (like the real
    arch_prop/typeacc_method instances the registry stores) and carrying a
    ``.func`` so the unowned-set error message can format."""

    def __init__(self, func=lambda: None):
        self.func = func


def test_claim_then_get_fields_preserves_order():
    reg = FieldRegistry()

    class Owner: ...

    a, b = _Descriptor(), _Descriptor()
    reg.mark_unowned(a)
    reg.mark_unowned(b)
    reg.claim(Owner, Cat, "x", a, singular=False)
    reg.claim(Owner, Cat, "y", b, singular=False)

    assert reg.get_fields(Owner, Cat) == ["x", "y"]
    # an unrelated descriptor type on the same owner is empty, not an error
    assert reg.get_fields(Owner, OtherCat) == []


def test_unknown_owner_returns_empty():
    reg = FieldRegistry()

    class Owner: ...

    assert reg.get_fields(Owner, Cat) == []


def test_singular_overwrite_raises():
    reg = FieldRegistry()

    class Owner: ...

    a, b = _Descriptor(), _Descriptor()
    reg.mark_unowned(a)
    reg.mark_unowned(b)
    reg.claim(Owner, Cat, "first", a, singular=True)
    with pytest.raises(AttributeError, match="singular"):
        reg.claim(Owner, Cat, "second", b, singular=True)


def test_duplicate_name_in_bucket_raises():
    reg = FieldRegistry()

    class Owner: ...

    a, b = _Descriptor(), _Descriptor()
    reg.mark_unowned(a)
    reg.mark_unowned(b)
    reg.claim(Owner, Cat, "dup", a, singular=False)
    with pytest.raises(AttributeError, match="unique"):
        reg.claim(Owner, Cat, "dup", b, singular=False)


def test_unowned_descriptor_blocks_lookups_until_claimed():
    reg = FieldRegistry()

    class Owner: ...

    leaked = _Descriptor()
    reg.mark_unowned(leaked)
    with pytest.raises(AttributeError, match="have not been owned"):
        reg.get_fields(Owner, Cat)

    # claiming it clears the unowned set and lookups work again
    reg.claim(Owner, Cat, "x", leaked, singular=False)
    assert reg.get_fields(Owner, Cat) == ["x"]


def test_claim_without_mark_unowned_fails_loud():
    # `claim` removes the descriptor from the unowned set; if it was never
    # registered there the invariant (mark_unowned in __init__ precedes
    # claim in __set_name__) is broken and we want a loud failure, not a
    # silent one. This is why claim uses set.remove, not set.discard.
    reg = FieldRegistry()

    class Owner: ...

    with pytest.raises(KeyError):
        reg.claim(Owner, Cat, "x", _Descriptor(), singular=False)


def test_get_from_fields_returns_dict_when_not_singular():
    reg = FieldRegistry()

    class Owner:
        a = "VA"
        b = "VB"

    da, db = _Descriptor(), _Descriptor()
    reg.mark_unowned(da)
    reg.mark_unowned(db)
    reg.claim(Owner, Cat, "a", da, singular=False)
    reg.claim(Owner, Cat, "b", db, singular=False)

    assert reg.get_from_fields(Owner(), Cat, singular=False) == {"a": "VA", "b": "VB"}


def test_get_from_fields_returns_value_when_singular():
    reg = FieldRegistry()

    class Owner:
        only = "ONE"

    d = _Descriptor()
    reg.mark_unowned(d)
    reg.claim(Owner, Cat, "only", d, singular=True)

    assert reg.get_from_fields(Owner(), Cat, singular=True) == "ONE"


def test_fields_are_found_through_inheritance():
    reg = FieldRegistry()

    class Base: ...

    class Sub(Base): ...

    d = _Descriptor()
    reg.mark_unowned(d)
    reg.claim(Base, Cat, "x", d, singular=False)

    # a subclass that declares no fields of this type inherits the base's
    assert reg.get_fields(Sub, Cat) == ["x"]


def test_subclass_fields_merge_with_base():
    reg = FieldRegistry()

    class Base: ...

    class Sub(Base): ...

    base_d, sub_d = _Descriptor(), _Descriptor()
    reg.mark_unowned(base_d)
    reg.mark_unowned(sub_d)
    reg.claim(Base, Cat, "base_field", base_d, singular=False)
    reg.claim(Sub, Cat, "sub_field", sub_d, singular=False)

    # merged base-first across the MRO: the subclass sees the base's fields
    # followed by its own (rather than shadowing the base's)
    assert reg.get_fields(Sub, Cat) == ["base_field", "sub_field"]
    assert reg.get_fields(Base, Cat) == ["base_field"]


def test_subclass_redeclared_name_appears_once():
    # an override (same name declared on the subclass) is deduped: getattr
    # resolves the name to the subclass's version anyway, so the merged list
    # carries the name a single time rather than tripping a duplicate.
    reg = FieldRegistry()

    class Base: ...

    class Sub(Base): ...

    base_d, sub_d = _Descriptor(), _Descriptor()
    reg.mark_unowned(base_d)
    reg.mark_unowned(sub_d)
    reg.claim(Base, Cat, "shared", base_d, singular=False)
    reg.claim(Sub, Cat, "shared", sub_d, singular=False)

    assert reg.get_fields(Sub, Cat) == ["shared"]


def test_distinct_descriptor_types_do_not_collide():
    reg = FieldRegistry()

    class Owner: ...

    a, b = _Descriptor(), _Descriptor()
    reg.mark_unowned(a)
    reg.mark_unowned(b)
    reg.claim(Owner, Cat, "x", a, singular=False)
    reg.claim(Owner, OtherCat, "y", b, singular=False)

    assert reg.get_fields(Owner, Cat) == ["x"]
    assert reg.get_fields(Owner, OtherCat) == ["y"]
