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
from saeco.evaluation.evaluation_context import Evaluation
import nnsight

from rich.highlighter import Highlighter

# from transformers import GPT2LMHeadModel
ec = Evaluation.from_cache_name("ec_test")
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

tangent = torch.zeros(1, 10, 6144).cuda()
tangent[0, 3, :] = 1
i = torch.arange(10).reshape(2, 5)
i[0, 3] += 2000
ec.saved_acts.tokens[i].shape
ec.saved_acts.tokens[torch.arange(6).unsqueeze(0)].shape
ec.saved_acts.tokens[0:5, 0:5].shape


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


print(out1)
print(out1.logits - out2.logits)
print(tangent)
# %%

import torch.autograd.forward_ad as fwAD


def fwad_hook(acts):
    tangent = torch.zeros_like(acts)
    tangent[0, 3, :] = 1

    return fwAD.make_dual(acts, tangent)


patched_sae = ec.sae_with_patch(fwad_hook)
with fwAD.dual_level():
    with nnsight_model.trace("The Eieffel Tower is in the city of") as tracer:

        lm_acts = getsite(nnsight_model, nn_name)
        orig_lm_acts = lm_acts.save()
        acts_re = patched_sae(orig_lm_acts).save()
        setsite(nnsight_model, nn_name, acts_re)
        out = nnsight_model.output.save()
        tangent = nnsight.apply(fwAD.unpack_dual, out.logits).tangent.save()

out1 = out.value.logits
print(out1)
print(tangent)


ec.saved_acts.tokens[0:5]
# %%

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

ec.saved_acts.tokens[torch.arange(5).long().unsqueeze(1)]
ec.saved_acts.acts[torch.arange(5).long().unsqueeze(0)]
# %%
ec.saved_acts.chunks[0].acts.values
# %%
