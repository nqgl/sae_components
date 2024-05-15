from typing import Any, Protocol, List, Callable, Union, runtime_checkable
from functools import wraps


@runtime_checkable
class HasName(Protocol):
    _default_component_name: str


class ProvidedReq:
    def __init__(self, provided, s=None):
        self.provided = provided
        self._default_component_name = s

    def get(self):
        return self.provided


class ProvidedReqGetter:
    provided: Callable

    def __init__(self, provided, cls_provided, s=None):
        self.provided = provided
        self.cls_provided = cls_provided
        self.s = s


class Req:
    def __init__(self, cls):
        self.req = cls

    def __call__(self, candidate) -> Any:
        if isinstance(self.req, str):
            if isinstance(candidate, HasName):
                return self.req == candidate._default_component_name
            else:
                raise ValueError(
                    f"Requirement {self.req} is a string but candidate does not have a name"
                )
        elif isinstance(candidate, ProvidedReqGetter):
            return issubclass(candidate.cls_provided, self.req)
        return isinstance(candidate, self.req)


class RequiredReq:
    def __init__(self, req: Req):
        if not isinstance(req, Req):
            req = Req(req)
        self.provided_req = None
        self.req = req

    def provide(self, prov: ProvidedReq):
        if self.req(prov):
            assert self.provided_req is None
            self.provided_req = prov

    def get_req(self):
        if self.provided_req is None:
            raise ValueError(f"Requirement {self.req} accessed but not fulfilled")
        return self.provided_req.get()

    def fulfilled(self):
        return self.provided_req is not None


@runtime_checkable
class CanRequire(Protocol):
    _requires: List[RequiredReq]


def required_field(obj: CanRequire, req: Req):
    rr = RequiredReq(req)
    obj._requires.append(rr)
    return property(rr.get_req)


@runtime_checkable
class CanProvide(Protocol):
    _provides: List[ProvidedReq]


def provided_field(obj: CanProvide, provided: Callable, s=None):
    pr = ProvidedReq(provided, s)
    obj._provides.append(pr)
    return provided


def provided_getter(cls_provided, s=None):
    def decorator(provided):
        return ProvidedReqGetter(provided, cls_provided, s)

    return decorator


def provide(recv_obj: CanRequire, prov: ProvidedReq):
    for req in recv_obj._requires:
        req.provide(prov)


class ReqCollection:
    def __init__(self, items: List[Union[CanProvide, CanRequire]]):
        self._requires = []
        self._provides = []
        for item in items:
            if isinstance(item, CanRequire):
                self._requires += item._requires
        for item in items:
            if isinstance(item, CanProvide):
                self._provides += item._provides
        for prov in self._provides:
            for req in self._requires:
                req.provide(prov)

    def finalize(self):
        pass


def main():
    class TestRP1:
        def __init__(self):
            self._provides = []
            self._requires = []
            self.a_req = required_field(self, int)
            self.s = "abc"

        @required_field(int)
        def a_req(self): ...

        def ret_str(self):
            return "ABC"

        @provided_getter(str)
        def ret_dynamic_str(self):
            return self.s

    class IntProvider:
        def __init__(self):
            self._provides = []
            self.an_int = provided_field(self, 5)

    t1 = TestRP1()
    ip = IntProvider()

    rc = ReqCollection([t1, ip])
    print(t1.a_req)
    print(t1.prop_test)


if __name__ == "__main__":
    main()


"""
    things can require a default cache 


"""
