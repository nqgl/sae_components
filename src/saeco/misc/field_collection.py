from collections import defaultdict
from typing import Any


class FieldRegistry:
    """Composed collector for "type-accumulated" class members.

    A descriptor (e.g. ``arch_prop`` / ``typeacc_method``) that wants to be
    discoverable *by its own type* registers itself here from its
    ``__set_name__``. The registry records, per owning class, the attribute
    names declared with each descriptor type, so the framework can later
    enumerate "every member of this class declared with descriptor type X" —
    walking the owner's MRO so inherited declarations are found too.

    This object owns all the bookkeeping; descriptor classes merely hold a
    reference and forward to it (composition, not inheritance). It is access
    agnostic and never touches ``__get__`` / ``__call__`` — *how* a field is
    read back stays entirely with the descriptor.
    """

    def __init__(self) -> None:
        # owner class -> (descriptor type -> attribute names declared on it)
        self._fields: dict[type, dict[type, list[str]]] = defaultdict(dict)
        # descriptors constructed but not yet bound to a class attribute
        self._unowned: set[Any] = set()

    def mark_unowned(self, descriptor: Any) -> None:
        """Record a freshly-constructed descriptor (call from its ``__init__``).

        Cleared when the descriptor is bound to a name via :meth:`claim`. A
        non-empty set at lookup time means some descriptor was built but never
        assigned as a class attribute, which would silently drop it.
        """
        self._unowned.add(descriptor)

    def claim(
        self,
        owner: type,
        descriptor_type: type,
        name: str,
        descriptor: Any,
        *,
        singular: bool,
    ) -> None:
        """Bind ``name`` on ``owner`` as a field of ``descriptor_type``.

        Call from the descriptor's ``__set_name__``. Enforces that a singular
        field is not overwritten and that names within an
        ``(owner, descriptor_type)`` bucket are unique. ``descriptor`` must
        already be registered as unowned (it is, via :meth:`mark_unowned` from
        ``__init__``); a single instance is claimed exactly once, so a missing
        entry here means the invariant broke (e.g. one descriptor bound to two
        names) and should fail loudly.
        """
        self._unowned.remove(descriptor)
        owner_fields = self._fields[owner]
        if descriptor_type in owner_fields:
            if singular:
                raise AttributeError(
                    f"{descriptor_type}: Cannot overwrite singular field "
                    f"'{name}' on {owner}"
                )
            names = owner_fields[descriptor_type]
            names.append(name)
            if len(names) != len(set(names)):
                raise AttributeError(
                    f"{descriptor_type}: Field names must be unique: duplicate "
                    f"name '{name}' on {owner}"
                )
        else:
            owner_fields[descriptor_type] = [name]

    def get_fields(self, owner: type, descriptor_type: type) -> list[str]:
        """Names of ``descriptor_type`` declared on ``owner``, including inherited.

        Declarations are merged across the owner's MRO base-first, so a subclass
        sees its bases' fields followed by its own. A name redeclared on a
        subclass (an override) appears once; ``getattr`` resolves it to the
        subclass's version as usual.
        """
        if self._unowned:
            raise AttributeError(
                "some properties have not been owned: "
                f"{[d.func for d in self._unowned]}"
            )
        if not isinstance(owner, type):
            owner = type(owner)
        names: list[str] = []
        for klass in reversed(owner.__mro__):
            bucket = self._fields.get(klass)
            if bucket is None:
                continue
            for name in bucket.get(descriptor_type, ()):
                if name not in names:
                    names.append(name)
        return names

    def get_from_fields(
        self, inst: object, descriptor_type: type, *, singular: bool
    ) -> Any:
        """Resolve the field(s) of ``descriptor_type`` on a live instance.

        Returns the single attribute value for a singular descriptor type,
        otherwise a ``{name: value}`` dict.
        """
        fields = self.get_fields(type(inst), descriptor_type)
        if singular:
            assert len(fields) == 1
            return getattr(inst, fields[0])
        return {name: getattr(inst, name) for name in fields}


# A single registry shared by every type-accumulated descriptor family. Buckets
# are keyed by ``(owner, descriptor_type)`` and the families' descriptor types
# are disjoint, so sharing one instance never collides; it also keeps the
# "every collected descriptor got named" invariant a single global check.
FIELDS = FieldRegistry()
