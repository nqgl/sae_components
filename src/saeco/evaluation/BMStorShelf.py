from __future__ import annotations

import shelve
from pathlib import Path
from typing import Any, Self

from attrs import define, field

from .cache_keys import call_cache_key


@define(slots=True)
class BMStorShelf:
    """
    Small versioned cache on top of `shelve`.

    We store:
      - key  -> value
      - vkey -> version tag for that key

    Version comes from `func._version` (set via @src/saeco/evaluation/cache_version.py).
    """

    path: Path
    shelf: shelve.Shelf = field(repr=False)

    @classmethod
    def from_path(cls, path: Path) -> Self:
        path.mkdir(parents=True, exist_ok=True)
        return cls(path=path, shelf=shelve.open(str(path / "shelf")))

    def close(self) -> None:
        try:
            self.shelf.close()
        except Exception:
            # Shelve backends differ; closing should never crash callers.
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def version(func: Any) -> Any | None:
        return getattr(func, "_version", None)

    def _keys(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[str, str]:
        base = call_cache_key(
            func,
            args=args,
            kwargs=kwargs,
            version=None,  # version stored separately for easy invalidation
            prefix=getattr(func, "__qualname__", getattr(func, "__name__", "call")),
        )
        return base, f"{base}::version"

    def _legacy_keys(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[str, str]:
        base = f"{func.__name__}__{args}__{kwargs}"
        return base, f"{base}__version"

    def has(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        ver = self.version(func)
        key, vkey = self._keys(func, args, kwargs)
        if key in self.shelf and vkey in self.shelf and self.shelf[vkey] == ver:
            return True

        # Legacy fallback (older key scheme).
        lkey, lvkey = self._legacy_keys(func, args, kwargs)
        return lkey in self.shelf and lvkey in self.shelf and self.shelf[lvkey] == ver

    def get(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        key, _ = self._keys(func, args, kwargs)
        if key in self.shelf:
            return self.shelf[key]
        lkey, _ = self._legacy_keys(func, args, kwargs)
        return self.shelf[lkey]

    def set(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], value: Any
    ) -> None:
        key, vkey = self._keys(func, args, kwargs)
        self.shelf[key] = value
        self.shelf[vkey] = self.version(func)
        self.shelf.sync()