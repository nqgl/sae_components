# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
from saeco.evaluation.nnsite import getsite, setsite, tlsite_to_nnsite
from saeco.trainer import Trainable

from saeco.architectures.anth_update import cfg, anth_update_model

from jaxtyping import Int, Float
from torch import Tensor
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
from functools import wraps
from saeco.evaluation.evaluation import Evaluation
import nnsight

from rich.highlighter import Highlighter

# from transformers import GPT2LMHeadModel
ec = Evaluation.from_cache_name("dyn_thresh")
# ec = Evaluation.from_model_name("sae sweeps/dyn_thresh/50001")
# ec.store_acts(
#     caching_cfg=CachingConfig(
#         docs_per_chunk=1000,
#         num_chunks=10,
#         dirname="dyn_thresh",
#     ),
#     displace_existing=True,
# )
# %%
nnsight_model = nnsight.LanguageModel("openai-community/gpt2", device_map="cuda")

# %%
import tqdm


import einops

ec.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site


tl_name = "blocks.6.hook_resid_pre"
nn_name = tlsite_to_nnsite(tl_name)


# %%\


# %%
cache = ec.sae.make_cache()
cache.acts = ...

cache = ec.sae.make_cache()
cache.acts = ...


# def run_sae(lm_acts):
# # %%
# cache2 = ec.sae.make_cache()


# def run_sae2(lm_acts):
#     return ec.sae(lm_acts, cache=cache2)


# with nnsm.trace("The Eiffel Tower is in the city of") as tracer:
#     lm_acts =getsite(nnsm, nn_name)
#     orig_lm_acts = lm_acts.save()
#     acts_re = nnsight.apply(run_sae2, lm_acts).save()
#     out = nnsm.output.save()
# out2 = out.value.logits


# print(out1 - out2)


# # %%
# # TESTING FWAD

# # cache.register_write_callback()


# def patched_acts(patch_fn, give_cache=False):
#     def acts_hook(cache: TrainCache, acts, act_name=None):
#         """
#         Act_name=None corresponds to the main activations, usually correct
#         """
#         if cache._parent is None:
#             return acts
#         if cache.act_metrics_name is not act_name:
#             return acts
#         return patch_fn(acts, cache=cache) if give_cache else patch_fn(acts)

#     def call_sae(x):
#         cache = ec.sae.make_cache()
#         cache.acts = ...
#         cache.act_metrics_name = ...
#         cache.register_write_callback("acts", acts_hook)
#         return ec.sae(x, cache=cache)

#     return call_sae


# def fwad_hook(acts):
#     return fwAD.make_dual(acts, tangent)


# fwad_run = patched_acts(fwad_hook)

# with fwAD.dual_level():
#     with nnsm.trace("The Eiffel Tower is in the city of") as tracer:
#         lm_acts =getsite(nnsm, nn_name)
#         orig_lm_acts = lm_acts.save()
#         acts_re = nnsight.apply(fwad_run, lm_acts).save()

#         setsite(nnsm, nn_name, acts_re)
#         out = nnsm.output.save()
#     tangent = fwAD.unpack_dual(out.value.logits).tangent

# out1 = out.value.logits
# print(out1)
# # %%

# tangent = torch.zeros(1, 10, 6144).cuda()
# tangent[0, 3, :] = 1
# i = torch.arange(10).reshape(2, 5)
# i[0, 3] += 2000
# # ec.saved_acts.tokens[i].shape
# ec.saved_acts.tokens[torch.arange(6).unsqueeze(0)].shape
# ec.saved_acts.tokens[0:5, 0:5].shape


def patch_hook(acts):
    return acts
    # return nnsight.apply(fwAD.make_dual, acts, tangent)


with nnsight_model.trace("The Eieffel Tower is in the city of") as tracer:

    patched_sae = ec.sae_with_patch(patch_hook, for_nnsight=False)

    lm_acts = getsite(nnsight_model, nn_name)
    orig_lm_acts = lm_acts.save()
    acts_re = nnsight.apply(patched_sae, orig_lm_acts).save()
    setsite(nnsight_model, nn_name, acts_re)
    out1 = nnsight_model.output.save()


def patch_hook(acts):
    return torch.zeros_like(acts)


patched_sae = ec.sae_with_patch(patch_hook)
with nnsight_model.trace("The Eieffel Tower is in the city of") as tracer:
    lm_acts = getsite(nnsight_model, nn_name)
    orig_lm_acts = lm_acts.save()
    acts_re = patched_sae(orig_lm_acts).save()
    setsite(nnsight_model, nn_name, acts_re)
    out2 = nnsight_model.output.save()


# print(out1)
# print(out1.logits - out2.logits)
# print(tangent)
# %%

import torch.autograd.forward_ad as fwAD


def active(document, position):
    return ec.saved_acts.acts[document][position]


active(4, 5)

ec.saved_acts.tokens[0:5]
# %%


def fwad_hook(acts):
    return fwAD.make_dual(acts, acts)


patched_sae = ec.sae_with_patch(fwad_hook)
with fwAD.dual_level():
    with nnsight_model.trace(ec.saved_acts.tokens[0:5]) as tracer:

        lm_acts = getsite(nnsight_model, nn_name)
        orig_lm_acts = lm_acts.save()
        acts_re = patched_sae(orig_lm_acts).save()
        setsite(nnsight_model, nn_name, acts_re)
        out = nnsight_model.output.save()
        tangent = nnsight.apply(fwAD.unpack_dual, out.logits).tangent.save()
# %%

acts_list = []


def grad_hook(acts: Tensor):
    acts.retain_grad()
    acts_list.append(acts)
    return acts


tokens = ec.saved_acts.tokens[0:5]
with nnsight_model.trace(tokens) as tracer:
    lm_acts = getsite(nnsight_model, nn_name)
    orig_lm_acts = lm_acts.save()
    acts_re = ec.sae_with_patch(grad_hook)(orig_lm_acts).save()
    setsite(nnsight_model, nn_name, acts_re)
    out = nnsight_model.output
    logits122 = out.logits[:, 122, tokens[:, 122]]
    logits122.sum().backward()


# %%
acts_list2 = []


def grad_hook(acts: Tensor):
    # acts.retain_grad()
    # acts_list2.append(acts)
    return acts


tokens = ec.saved_acts.tokens[0:5]
grads = []
ec.sae.training = False
with nnsight_model.trace(tokens) as tracer:
    lm_acts = getsite(nnsight_model, nn_name)
    orig_lm_acts = lm_acts.save()
    res = ec.sae_with_patch(grad_hook, return_sae_acts=True)(orig_lm_acts)
    setsite(nnsight_model, nn_name, res[0])
    out = nnsight_model.output.save()
    for i in range(tokens.shape[1]):
        grads.append(res[1].grad.save())
        res[1].grad = None
        out.logits[torch.arange(5), i, tokens[torch.arange(5), i]].sum().backward(
            retain_graph=True
        )
    # logits122.sum().backward(retain_graph=True)
    # grads.save()
    # for logit in logits122:

    # torch.autograd.grad()
# torch.ones().backward(,

# %%
# ograds=grads

# (grads[2] == ograds[0]).all()
# # %%
# (grads[4] == grads[2]).all()

# len(grads)
# grads
# # %%

# sae_acts_grad
# acts_list2[0].grad.shape
# # %%
# acts_list[0].grad.shape

# # %%

# # %%
