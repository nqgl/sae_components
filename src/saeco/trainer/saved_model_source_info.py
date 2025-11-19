import importlib
import inspect
from warnings import deprecated

from pydantic import BaseModel


def get_src(fn):
    module = importlib.import_module(fn.__module__)
    return inspect.getsource(module)


@deprecated(
    "Deprecated as part of migration to Architecture. ArchReloadInfo replaces this functionality"
)
class ModelReloadInfo(BaseModel):
    module: str
    fn_name: str
    source_backup: str = None
    cfg_name: str = "cfg"

    @classmethod
    def from_model_fn(cls, model_fn: callable) -> "ModelReloadInfo":
        return cls(
            module=model_fn.__module__,
            fn_name=model_fn.__name__,
            source_backup=get_src(model_fn),
        )

    def get_model_and_cfg(self):
        module = importlib.import_module(self.module)
        model_fn = getattr(module, self.fn_name)
        if get_src(model_fn) != self.source_backup:
            print(
                """
                warning: loaded model source code appears to have changed since the model was saved. 
                This may cause issues.
                (but not necessarily)
                """
            )
        return model_fn, getattr(module, self.cfg_name)
