# %%
from saeco import core as cl
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 10),
    ),
)


def forward_pre_hook(module, a, k):
    print("forward_pre_hook:", module)
    print("args:", a)
    print("kwargs:", k)


model.register_forward_pre_hook


# %%
class TakesExtras(nn.Module):
    def forward(self, x, extra, akwarg):
        return x


class Model(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.extra = TakesExtras()
        self.extra.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)

    def forward(self, x):
        return self.extra(self.m(x), "extra", akwarg="akwarg")


[m for m in model.named_modules()]


exmodel = Model(model)
exmodel(torch.randn(2, 10))
# %%
model.apply


class CacheWrapper(cl.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __repr__(self):
        return self.model.__repr__()


def cachewrap_forward(model, name):
    old_forward = model.forward

    def new_forward(self, *args, cache, **kwargs):
        out = old_forward(*args, **kwargs)


model.__repr__


def modify(model: nn.Module):
    model.modules()
    modified = {}
    for m in model.modules():
        ...


# %%
def addkwarghook(module, a, k):
    print("forward_pre_hook:", module)
    print("args:", a)
    print("kwargs:", k)
    k["test_kwarg"] = "test_kwarg"


class TestModule(nn.Module):
    def forward(self, x, **kwargs):
        print("tm got kwargs:", kwargs)
        return x


tm = TestModule()
tm.register_forward_pre_hook(addkwarghook, with_kwargs=True)
tm(torch.randn(2, 2))


def usecache(model: nn.Module, cache: cl.Cache):
    # def hook_at_site()
    def cachekwarghook(module, a, k):
        if "cache" in k:
            assert k["cache"] is cache

    # for name, module in model.named_modules(,):
    #     module.register_forward_pre_hook(cachekwarghook, with_kwargs=True)


# %%

from collections import OrderedDict

relin = nn.Linear(10, 10)
model = nn.Sequential(
    OrderedDict(
        [
            ("l0", relin),
            ("l1", nn.ReLU()),
            ("l2", nn.Linear(10, 10)),
            ("l3", relin),
            ("l4", nn.ReLU()),
        ]
    )
)

from nnsight import NNsight

nmodel = NNsight(model)
ones = torch.ones(2, 10)
with nmodel.trace(torch.randn(2, 10)) as trace:
    nmodel.l0.output = ones
    out = nmodel.output.save()
    l0_out = nmodel.l0.output.save()
    l1_in = nmodel.l1.input.save()

# %%

l0_out.value
l1_in


out.value
# %%


class EdgeCase(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return self.a(x)


model = EdgeCase(nn.Linear(10, 10), nn.Linear(10, 10))

nmodel = NNsight(model)
with nmodel.trace(torch.randn(2, 10)) as trace:
    nmodel.a.output = torch.ones(2, 10)
    ao = nmodel.a.output.save()
    nmodel.b.input = ((torch.ones(2, 10),), {})
    # bi=nmodel.b.input.save()
    bo = nmodel.b.output.save()
    out = nmodel.output.save()
    ao2 = nmodel.a.output.save()

out
# %%
bo
# %%
bi.value
# %%
ao
# %%
ao2
# %%
type(nmodel)
# %%
