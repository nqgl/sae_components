import inspect
import types
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import (
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from saeco.core.cache import Cache

type HookWithoutCache[OwnerT] = Callable[[OwnerT], Any]
type HookWithCache[OwnerT, CacheT: Cache] = Callable[[OwnerT, CacheT], Any]
type HookFunction[OwnerT, CacheT: Cache] = (
    HookWithoutCache[OwnerT] | HookWithCache[OwnerT, CacheT]
)
type HookDecorator[HookT: "saeco_hook[Any, ...]", OwnerT, CacheT: Cache] = Callable[
    [HookFunction[OwnerT, CacheT]], HookT
]

_fields_dict: dict[type, dict[type, list[str]]] = defaultdict(dict)
# (cls -> (field_categ_name -> field_name/names))
_missing_name: set["typeacc_method"] = set()


def _getfields(cls: type, field_name: type) -> list[str]:
    if not isinstance(cls, type):
        cls = type(cls)
    cls_d = _fields_dict[cls]
    if field_name not in cls_d:
        for c in cls.__mro__:
            if c == cls or c is object:
                continue
            try:
                return _getfields(c, field_name)
            except AttributeError:
                pass
        raise AttributeError(field_name)
    return cls_d[field_name]


def getfields(cls: type, field_name: type) -> list[str]:
    try:
        return _getfields(cls, field_name)
    except AttributeError:
        return []


def setfield(cls: type, field_name: type, value: list[str]):
    assert isinstance(cls, type)
    if cls not in _fields_dict:
        _fields_dict[cls] = {}
    cls_d = _fields_dict[cls]
    cls_d[field_name] = value


def hasfield(cls: type, field_name: type):
    if cls not in _fields_dict:
        return False
    cls_d = _fields_dict[cls]
    return field_name in cls_d


class typeacc_method[T, **P]:  # noqa: N801  # decorator API; lowercase by decorator convention
    COLLECTED_FIELD_SINGULAR = False

    def __init__(self, func: Callable[P, T]) -> None:
        _missing_name.add(self)
        self.func = func
        self._name = None
        # update_wrapper(self, func)

    def __get__(self, instance, owner):
        return types.MethodType(self, instance)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    def __set_name__(self, owner: type, name: str) -> None:
        _missing_name.remove(self)
        self._name = name
        if hasfield(owner, self.__class__):
            if self.COLLECTED_FIELD_SINGULAR:
                raise AttributeError(
                    f"{self.__class__}: Cannot overwrite singular field '{name}' on "
                    f"{owner}"
                )
            fields = getfields(owner, self.__class__)
            fields.append(name)
            if len(fields) != len(set(fields)):
                raise AttributeError(
                    f"{self.__class__}: Field names must be unique: duplicate name "
                    f"'{name}' on {owner}"
                )
        else:
            setfield(owner, self.__class__, [name])

    @classmethod
    def get_fields(cls, owner: type):
        if len(_missing_name) > 0:
            raise AttributeError(
                "some properties have not been owned: "
                f"{[f.func for f in _missing_name]}"
            )
        return getfields(owner, cls)

    @classmethod
    def get_from_fields(cls, inst: object) -> dict[str, Callable[..., Any]]:
        fields = cls.get_fields(type(inst))
        assert not cls.COLLECTED_FIELD_SINGULAR
        return {f: getattr(inst, f) for f in fields}


def _resolved_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(func, include_extras=True)
    except (AttributeError, NameError, TypeError):
        return {}


def _cache_types_from_annotation(annotation: Any) -> tuple[type[Cache], ...]:
    if annotation is inspect.Signature.empty:
        return ()
    if isinstance(annotation, str):
        return ()
    if get_origin(annotation) in (Union, types.UnionType):
        cache_types = []
        for arg in get_args(annotation):
            cache_types.extend(_cache_types_from_annotation(arg))
        return tuple(dict.fromkeys(cache_types))
    if isinstance(annotation, type) and issubclass(annotation, Cache):
        return (annotation,)
    return ()


def _cache_parameter_and_types(
    func: Callable[..., Any], *, fallback: bool = False
) -> tuple[inspect.Parameter | None, tuple[type[Cache], ...]]:
    signature = inspect.signature(func)
    hints = _resolved_type_hints(func)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    if len(parameters) > 1:
        raise TypeError(
            f"{func.__qualname__} must be a hook method shaped like (self) or "
            "(self, cache)."
        )
    if not parameters:
        if fallback:
            raise TypeError(
                f"{func.__qualname__} was declared with takes_cache=True but has no "
                "cache parameter."
            )
        return None, ()

    parameter = parameters[0]
    if parameter.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(
            f"{func.__qualname__} must use a single explicit cache parameter, not "
            f"{parameter.kind.description}."
        )

    cache_types = _cache_types_from_annotation(
        hints.get(parameter.name, parameter.annotation)
    )
    if cache_types:
        return parameter, cache_types
    if fallback:
        if parameter.annotation is not inspect.Signature.empty:
            raise TypeError(
                f"{func.__qualname__} declares takes_cache=True but its cache "
                "parameter is not annotated as a Cache subtype."
            )
        return parameter, (Cache,)
    raise TypeError(
        f"{func.__qualname__} has one hook parameter, but it is not annotated as "
        "a Cache subtype and the hook decorator did not pass takes_cache=True."
    )


class saeco_hook[T, **P](typeacc_method[T, P]):  # noqa: N801
    def __init__(
        self, func: Callable[P, T], takes_cache: bool | None = None
    ) -> None:
        super().__init__(func)
        self.cache_parameter, self.cache_types = _cache_parameter_and_types(
            func, fallback=takes_cache is True
        )
        self.takes_cache = (
            self.cache_parameter is not None if takes_cache is None else takes_cache
        )

    def call(self, bound_hook: Callable[..., Any], *, cache: Any = None) -> Any:
        if not self.takes_cache:
            return bound_hook()
        if cache is None:
            raise ValueError(
                f"{bound_hook.__qualname__} requires cache, but no cache was "
                "provided."
            )
        if self.cache_types and not isinstance(cache, self.cache_types):
            accepted_types = ", ".join(t.__qualname__ for t in self.cache_types)
            raise TypeError(
                f"{bound_hook.__qualname__} requires cache of type {accepted_types}; "
                f"got {type(cache).__qualname__}."
            )
        parameter = self.cache_parameter
        if parameter is None or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return bound_hook(cache=cache)
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return bound_hook(cache)
        return bound_hook(**{parameter.name: cache})


class PreForwardHook[T, **P](saeco_hook[T, P]): ...


def _hook_decorator[HookT: saeco_hook[Any, ...], OwnerT, CacheT: Cache](
    hook_cls: type[HookT],
    f: HookFunction[OwnerT, CacheT] | None = None,
    *,
    takes_cache: bool | None = None,
) -> Any:
    def decorate(func: HookFunction[OwnerT, CacheT]) -> HookT:
        hook = hook_cls(cast(Callable[..., Any], func), takes_cache=takes_cache)
        return cast(HookT, wraps(cast(Callable[..., Any], func))(hook))

    if f is None:
        return decorate
    return decorate(f)


@overload
def pre_forward_hook[OwnerT](
    f: HookWithoutCache[OwnerT],
) -> PreForwardHook: ...
@overload
def pre_forward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT],
) -> PreForwardHook: ...
@overload
def pre_forward_hook[OwnerT](
    f: HookWithoutCache[OwnerT], *, takes_cache: Literal[False]
) -> PreForwardHook: ...
@overload
def pre_forward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT], *, takes_cache: Literal[True]
) -> PreForwardHook: ...
@overload
def pre_forward_hook[OwnerT](
    *, takes_cache: Literal[False]
) -> Callable[[HookWithoutCache[OwnerT]], PreForwardHook]: ...
@overload
def pre_forward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: Literal[True]
) -> Callable[[HookWithCache[OwnerT, CacheT]], PreForwardHook]: ...
@overload
def pre_forward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: None = None
) -> HookDecorator[PreForwardHook, OwnerT, CacheT]: ...
def pre_forward_hook[OwnerT, CacheT: Cache](
    f: HookFunction[OwnerT, CacheT] | None = None,
    *,
    takes_cache: bool | None = None,
) -> Any:
    return _hook_decorator(PreForwardHook, f, takes_cache=takes_cache)


class PostForwardHook[T, **P](saeco_hook[T, P]): ...


@overload
def post_forward_hook[OwnerT](
    f: HookWithoutCache[OwnerT],
) -> PostForwardHook: ...
@overload
def post_forward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT],
) -> PostForwardHook: ...
@overload
def post_forward_hook[OwnerT](
    f: HookWithoutCache[OwnerT], *, takes_cache: Literal[False]
) -> PostForwardHook: ...
@overload
def post_forward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT], *, takes_cache: Literal[True]
) -> PostForwardHook: ...
@overload
def post_forward_hook[OwnerT](
    *, takes_cache: Literal[False]
) -> Callable[[HookWithoutCache[OwnerT]], PostForwardHook]: ...
@overload
def post_forward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: Literal[True]
) -> Callable[[HookWithCache[OwnerT, CacheT]], PostForwardHook]: ...
@overload
def post_forward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: None = None
) -> HookDecorator[PostForwardHook, OwnerT, CacheT]: ...
def post_forward_hook[OwnerT, CacheT: Cache](
    f: HookFunction[OwnerT, CacheT] | None = None,
    *,
    takes_cache: bool | None = None,
) -> Any:
    return _hook_decorator(PostForwardHook, f, takes_cache=takes_cache)


class PostBackwardHook[T, **P](saeco_hook[T, P]): ...


@overload
def post_backward_hook[OwnerT](
    f: HookWithoutCache[OwnerT],
) -> PostBackwardHook: ...
@overload
def post_backward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT],
) -> PostBackwardHook: ...
@overload
def post_backward_hook[OwnerT](
    f: HookWithoutCache[OwnerT], *, takes_cache: Literal[False]
) -> PostBackwardHook: ...
@overload
def post_backward_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT], *, takes_cache: Literal[True]
) -> PostBackwardHook: ...
@overload
def post_backward_hook[OwnerT](
    *, takes_cache: Literal[False]
) -> Callable[[HookWithoutCache[OwnerT]], PostBackwardHook]: ...
@overload
def post_backward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: Literal[True]
) -> Callable[[HookWithCache[OwnerT, CacheT]], PostBackwardHook]: ...
@overload
def post_backward_hook[OwnerT, CacheT: Cache](
    *, takes_cache: None = None
) -> HookDecorator[PostBackwardHook, OwnerT, CacheT]: ...
def post_backward_hook[OwnerT, CacheT: Cache](
    f: HookFunction[OwnerT, CacheT] | None = None,
    *,
    takes_cache: bool | None = None,
) -> Any:
    return _hook_decorator(PostBackwardHook, f, takes_cache=takes_cache)


class PostStepHook[T, **P](saeco_hook[T, P]): ...


@overload
def post_step_hook[OwnerT](
    f: HookWithoutCache[OwnerT],
) -> PostStepHook: ...
@overload
def post_step_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT],
) -> PostStepHook: ...
@overload
def post_step_hook[OwnerT](
    f: HookWithoutCache[OwnerT], *, takes_cache: Literal[False]
) -> PostStepHook: ...
@overload
def post_step_hook[OwnerT, CacheT: Cache](
    f: HookWithCache[OwnerT, CacheT], *, takes_cache: Literal[True]
) -> PostStepHook: ...
@overload
def post_step_hook[OwnerT](
    *, takes_cache: Literal[False]
) -> Callable[[HookWithoutCache[OwnerT]], PostStepHook]: ...
@overload
def post_step_hook[OwnerT, CacheT: Cache](
    *, takes_cache: Literal[True]
) -> Callable[[HookWithCache[OwnerT, CacheT]], PostStepHook]: ...
@overload
def post_step_hook[OwnerT, CacheT: Cache](
    *, takes_cache: None = None
) -> HookDecorator[PostStepHook, OwnerT, CacheT]: ...
def post_step_hook[OwnerT, CacheT: Cache](
    f: HookFunction[OwnerT, CacheT] | None = None,
    *,
    takes_cache: bool | None = None,
) -> Any:
    return _hook_decorator(PostStepHook, f, takes_cache=takes_cache)
