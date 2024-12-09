from .fns import init, finish, enter, get_config
from typing import Callable, Any


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

    def setter(self, __fset: Callable[[Any, Any], None]) -> property:
        raise NotImplementedError

    def __set__(self, owner_self, value):
        raise NotImplementedError


class mlog:
    @staticmethod
    def init():
        init()

    @classproperty
    def config(cls):
        return get_config()

    @staticmethod
    def enter():
        return enter()

    @staticmethod
    def finish():
        return finish()

    @staticmethod
    def instantiate_config(cfg): ...
