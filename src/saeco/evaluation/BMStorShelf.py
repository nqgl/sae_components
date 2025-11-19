from attrs import define


import shelve
from pathlib import Path


@define
class BMStorShelf:
    path: Path
    shelf: shelve.Shelf

    @classmethod
    def from_path(cls, path: Path):
        return cls(
            path=path,
            shelf=shelve.open(str(path / "shelf")),
        )

    def fnkey(self, func, args, kwargs):
        key = f"{func.__name__}__{args}__{kwargs}"
        vkey = f"{key}__version"
        return key, vkey

    def version(self, func):
        return getattr(func, "_version", None)

    def has(self, func, args, kwargs):
        key, vkey = self.fnkey(func, args, kwargs)
        version = self.version(func)
        return vkey in self.shelf and self.shelf[vkey] == version and key in self.shelf

    def get(self, func, args, kwargs):
        key, vkey = self.fnkey(func, args, kwargs)
        return self.shelf[key]

    def set(self, func, args, kwargs, value):
        key, vkey = self.fnkey(func, args, kwargs)
        self.shelf[key] = value
        self.shelf[vkey] = self.version(func)
        self.shelf.sync()
