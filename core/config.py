from typing import Any, Set
import wandb
from dataclasses import dataclass, field


@dataclass
class WandbDynamicConfig:
    _dontlog: Set[str] = field(default_factory=set)
    _name: str = None
    _fullname: str = None
    _done_init: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._name is None:
            self._name = self.__class__.__name__
        if self._fullname is None:
            self._fullname = self._name
        self._done_init = False

    def __post_init__(self):
        self.update_fullname(self._fullname)
        self._done_init = True

    def update_fullname(self, parent_name):
        self._fullname = f"{parent_name}/{self._name}"
        for child in self.__dict__.values():
            if isinstance(child, WandbDynamicConfig):
                child.update_fullname(self._fullname)

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        object.__setattr__(self, __name, __value)
        if (
            wandb.run
            and not __name.startswith("_")
            and __name not in self._dontlog
            and self._done_init
        ):
            wandb.log({f"{self._fullname}/{__name}": __value}, step=wandb.run.step)
