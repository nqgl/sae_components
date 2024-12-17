from pydantic import BaseModel

import torch
import inspect
import importlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .architecture import Architecture


# def get_src(fn):
#     module = importlib.import_module(fn.__module__)
#     return inspect.getsource(module)


def get_src(obj):
    module = importlib.import_module(obj.__module__)
    return inspect.getsource(module)


class ArchReloadInfo(BaseModel):
    module: str
    cls_name: str
    source_backup: str = None

    @classmethod
    def from_arch(cls, arch: "Architecture") -> "ArchReloadInfo":
        return cls(
            module=arch.__class__.__module__,
            cls_name=arch.__class__.__name__,
            source_backup=get_src(arch.__class__),
        )

    def get_arch_class(self):
        module = importlib.import_module(self.module)
        arch_cls = getattr(module, self.cls_name)
        from .architecture import Architecture

        assert issubclass(arch_cls, Architecture)
        if get_src(arch_cls) != self.source_backup:
            print(
                """
                warning: loaded model source code appears to have changed since the model was saved. 
                This may cause issues.
                (but not necessarily)
                """
            )
        return arch_cls
