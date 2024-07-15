# %%
from saeco.sweeps import SweepableConfig
from typing import Any, Optional, overload, Callable
from functools import wraps
from pydantic import Field


@overload
def tosteps(n: int, period: None) -> Callable[[int | float], int]: ...


@overload
def tosteps(n: int | float, period: int) -> int: ...


def tosteps(n: int | float, period: int | None = None) -> str:
    # some values will be expressed as either
    # a number of steps
    # or a fraction of some period, default run length
    # signified by type -- ints are steps, floats are proportions
    # this converts proportions to steps and leaves steps as is
    assert 0 <= n
    if period is None:
        period = n

        @wraps(tosteps)
        def inner(n: int | float) -> int:
            return tosteps(n, period)

        return inner
    if isinstance(n, int):
        return n
    assert isinstance(n, float) and n <= 1 and isinstance(period, int)
    return n * period


#     def __init__(self, raw: "RunSchedulingConfig"):
#         self.raw = raw


#     def __getattribute__(self, name: str) -> Any:

# # PERIODS:
# # run_length
# # resample_period


# # resample_delay: int = 0

# # resampling_finished_phase
# # lr_warmup_length: int | float = 0.1
# # lr_cooldown_length: int | float = 0.2
# # lr_resample_warmup_length: Optional[int | float] = 0.2

# # targeting_post_resample_cooldown
# # targeting_resample_cooldown_period_override: Optional[int] = None
# # targeting_post_resample_hiatus: float | int = 0
# # targeting_delay


PERIODS = {
    "run_length": [
        "resample_delay",
        "lr_warmup_length",
        "lr_cooldown_length",
        "resampling_finished_phase",
        "targeting_delay",
    ],
    "resample_period": [
        "targeting_post_resample_cooldown",
        "targeting_post_resample_hiatus",
        "lr_resample_warmup_length",
    ],
}


AmbiguousTypes = [Optional[int | float], int | float]
# Run length
# ", resample period


class RunSchedulingConfig(SweepableConfig):
    run_length: Optional[int] = 5e3

    resample_period: int = 2_000
    resample_delay: int | float = 0
    resampling_finished_phase: int | float = 0

    targeting_post_resample_cooldown: int | float = 0.3
    # targeting_resample_cooldown_period_override: Optional[int] = None
    targeting_post_resample_hiatus: int | float = 0
    targeting_delay: int | float = 2000  # could be none -> copy cooldown

    ### lr scheduler # this is not quite the continuous pretraining scheduler, seems fine though
    lr_warmup_length: int | float = 0.1
    lr_cooldown_length: int | float = 0.2
    lr_resample_warmup_length: Optional[int | float] = 0.2
    lr_warmup_factor: float = 0.1
    lr_cooldown_factor: float = 0.1
    lr_resample_warmup_factor: float = 0.1

    # def model_post_init(self):

    #     for name, field in self.model_fields.items():

    @property
    def resample_start(self) -> int:
        return self.tosteps(self.resample_delay)

    @property
    def resample_end(self) -> int:
        return self.run_length - self.tosteps(self.resampling_finished_phase)

    def dynamic_adjust(self, t):
        if t < self.targeting_delay:
            return False
        rt = self.resample_t(t)
        if rt != -1 and rt < self.tosteps(
            self.targeting_post_resample_hiatus, self.resample_period
        ):
            return False
        return True

    def tosteps(self, n: int | float, period: int = None) -> int:
        # some values will be expressed as either
        # a number of steps
        # or a fraction of some period, default run length
        # signified by type -- ints are steps, floats are proportions
        # this converts proportions to steps and leaves steps as is
        assert 0 <= n
        if isinstance(n, int):
            return n
        assert isinstance(n, float) and n <= 1
        period = period or self.run_length
        return n * period

    def lr_scale(self, t: int) -> float:
        re_lr = 1
        if self.lr_resample_warmup_length and (rt := self.resample_t(t)) != -1:
            re_warmup = self.tosteps(
                self.lr_resample_warmup_length, self.resample_period
            )
            re_lr = max(min(rt / re_warmup, 1), self.lr_resample_warmup_factor)
        warmup = self.tosteps(self.lr_warmup_length)
        if t < warmup:
            return re_lr * max(t / warmup, self.lr_warmup_factor)
        to_end = self.run_length - t
        cooldown = self.tosteps(self.lr_cooldown_length)
        if to_end < cooldown:
            return re_lr * max(to_end / cooldown, self.lr_cooldown_factor)
        return re_lr

    def resample_t(self, t: int) -> int:
        if t < self.resample_delay:
            return -1
        if t - self.resample_delay + self.resample_period > self.resample_end:
            return -1
        return (t - self.resample_delay) % self.resample_period

    def is_resample_step(self, t: int) -> bool:
        return self.resample_t(t) == 0
        if t < self.resample_delay:
            return False
        if (t - self.resample_delay) % self.resample_period == 0:
            return True
        return False


rs = RunSchedulingConfig()
print(type(rs.resampling_finished_phase))


# %%
from pydantic import BaseModel, Field, ConfigDict, computed_field


def gen(t):
    print(t)
    return t + "k"


class T(BaseModel):
    bar: int
    dist: int = 3
    foo: int | float = Field(alias="boo")
    # bar: int = Field(alias="foo")

    # model_config = ConfigDict(populate_by_name=False)

    # @property
    # def bar(self) -> int:
    #     if isinstance(self.foo, int):
    #         return self.foo
    #     return 1 / self.foo

    # @property
    # def dist(self):
    #     return self.bar

    who: int = Field(alias="who2")

    @property
    def who(self):
        return self.foo

    @property
    def normal_prop(self):
        return 32

    def model_post_init(self, __context):
        print("post_init", __context)
        newfields = {}
        newdict = {}
        print(type(self.model_fields))

        # for name, field in self.model_fields.items():
        #     print(name, field)
        #     field.alias = name
        #     field.alias_priority = 2
        #     newfields[name + "_base"] = field
        #     # self.bar = self.bar + 1000
        #     value = super().__getattribute__
        #     @property
        #     def newfield():
        #         print(f"accessed {name}")
        #         return field
        #     newdict[name] = newfield

        # self.__dict__.update(newfields)
        # object.__setattr__(self,  "model_fields", newfields)
        # self.model_fields=newfields


# print(3)
t = T(bar=2, boo=69, who2=222)
# t=T(bar=2, boo=3)
print(t.model_dump())
t.bar_base
# %%
t.who, t, t.__dict__


t.__class__.__dict__.keys()


# %%
seen = set()


def search(d, depth=0):
    ret = False
    for k in d:
        sd = d[k]
        sk = (id(d), id(sd), id(k))
        if sk in seen:
            continue
        seen.add(sk)
        try:
            if "normal_prop" in sd or search(sd, depth=depth + 1):
                try:
                    print("    " * depth, k)
                except:
                    print("couldn't print key")
                ret = True
        except Exception as e:
            # print("e" ,k, e)
            pass
    return ret


search(t.__dict__)
# t.__class__.__dict__["__pydantic_parent_namespace__"]["_oh"]


# %%
from pydantic.version import version_info

print(version_info())
# %%

T.__dict__["normal_prop"]

c2 = type("c2", T.__bases__, dict(T.__dict__))


# %%

ll = []


def t2(c):
    def testdec(f):
        # print(ReasonableApproach2)
        print(c)
        print(f.__name__)
        ll.append(f)
        return f

    return testdec


class ReasonableApproach2(BaseModel):
    foo_v: int | float = Field(alias="foo")

    @property
    @t2(ReasonableApproach2)
    def foo(self):
        return tosteps(self.foo_v, 10)


i = ReasonableApproach(foo=340)
#
i.model_dump(by_alias=True)
# %%
i.model_fields
# %%
i.foo
# %%
f = ll[0]
# %%
dir(f)


# %%


def duh_class_decorator_try_again(cls: type[BaseModel]):
    class Class2:
        def __init__(self, *a, **k):
            self.cfg = cls(*a, **k)

        def model_dump(self, *a, **k):
            return self.cfg.model_dump(*a, **k, by_alias=True)

    mfi = {k: v for k, v in cls.model_fields.items()}
    for name, field in mfi.items():
        annotation = cls.__annotations__[name]
        if issubclass(int, annotation) and issubclass(float, annotation):
            print("found", name, field)
            field.alias = name
            # field.alias_priority = 2#?
            # newname = name + "_base"
            cls.model_fields.pop(name)
            # cls.model_fields[newname] = field
            cls.__private_attributes__

            def get_replacements(name):
                @property
                def replace_field(self: Class2):
                    value = getattr(self.cfg, name)
                    print("get", name, value)
                    if isinstance(value, int):
                        return -1 * value

                    return 23

                @replace_field.setter
                def replace_field(self: Class2, value):
                    print("set")
                    setattr(self.cfg, name, value)

                return replace_field

            setattr(Class2, name, get_replacements(name))
    # model_dump = cls.model_dump

    # def aliasdump(self, *a, **k):
    #     print(k)

    # cls.model_dump = aliasdump
    return Class2


def duh_class_decorator(cls: type[BaseModel]):

    mfi = {k: v for k, v in cls.model_fields.items()}
    for name, field in mfi.items():
        annotation = cls.__annotations__[name]
        if issubclass(int, annotation) and issubclass(float, annotation):
            print("found", name, field)
            field.alias = name
            # field.alias_priority = 2#?
            newname = name + "_base"
            cls.model_fields.pop(name)
            cls.model_fields[newname] = field
            cls.__private_attributes__

            @property
            def replace_field(self):
                value = getattr(self, newname)
                if isinstance(value, int):
                    return -1 * value

                return 23

            @replace_field.setter
            def replace_field(self, value):
                print("set")
                setattr(self, newname, value)

            setattr(cls, name, replace_field)
    model_dump = cls.model_dump

    def aliasdump(self, *a, **k):
        print(k)

    cls.model_dump = aliasdump
    return cls
    # def __init__(self, wrapped: BaseModel):
    #     self.cfg = wrapped

    #     @property
    #     def test(self):
    #         return 3

    #     self.test = test


@duh_class_decorator_try_again
class AClass(BaseModel):
    foo: int | float = 3
    normal: int = 3
    also_normal: float = 0.1


a = AClass(foo=20, normal=2)
# %%
a.foo, a.model_dump()
# %%
b = AClass(foo=0.2)
b.foo, b.model_dump()

# %%
