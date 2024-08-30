# %%
from pathlib import Path
from saeco.trainer import Trainable
from saeco.architectures.threshgate_gradjust.tg_grad_deep_model import (
    cfg,
    deep_tg_grad_sae,
)
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
import sys
import os
import torch

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from investigate import test

# %%
modelss = Path.home() / "workspace/saved_models/"


name = "sweep_None/(lars)deep_tg_grad_sae0.003[25.0]-31_10000"


def load(cfg: BaseModel, model_fn, name, modify_cfg_cb=None):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    if modify_cfg_cb is not None:
        modify_cfg_cb(cfg)
    tr = TrainingRunner(cfg, model_fn)
    sd = torch.load(
        pt_path,
    )
    print(tr.trainer.train)
    # for k in list(sd.keys()):
    #     if "mlp.proj_out.wrapped.wrapped" in k or "mlp.proj_in.wrapped.wrapped" in k:
    #         sd[k.replace("wrapped.wrapped", "wrapped.wrapped.wrapped")] = sd[k]
    #         del sd[k]
    # .replace("wrapped.wrapped", "wrapped.wrapped.wrapped")
    print("load result", tr.trainable.load_state_dict(sd))

    return tr


def acc(obj, name):
    if "." in name:
        name = name.split(".")
        for n in name:
            obj = acc(obj, n)
        return obj
    if name.isnumeric():
        return obj[int(name)]
    return getattr(obj, name)


def modify_cfg(cfg):
    cfg.arch_cfg.leniency = 1.4
    cfg.train_cfg.coeffs["sparsity_loss"] = 3e-4 * 0.03
    # cfg.train_cfg.use_averaged_model = True
    cfg.train_cfg.raw_schedule_cfg.run_length = 10_000


tr = load(cfg, deep_tg_grad_sae, name, modify_cfg_cb=modify_cfg)
model = tr.trainable
from saeco.trainer.recons import get_recons_loss

print(tr.trainer.train)


def model_call(x):
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float32,
    ):
        x = x.float()
        cache = tr.trainer.get_cache()
        cache.scale = ...
        cache.scale = True
        return tr.trainable(x, cache=cache)


cache = tr.trainer.get_cache()
tr.trainable.loss(next(tr.data), cache=cache)
print(cache.logdict())
print(tr.trainable.training)  # buf = tr.trainer.get_databuffer(num_workers=0)

print(
    "recons:",
    get_recons_loss(
        tr.trainer.subject_model,
        model_call,
        buffer=None,
        all_tokens=tr.trainer.llm_val_tokens,
        cfg=tr.trainer.cfg.data_cfg.model_cfg.acts_cfg,
        bos_processed_with_hook=False,
        num_batches=5,
    ),
)

model.eval()
print(
    "recons:",
    get_recons_loss(
        tr.trainer.subject_model,
        model_call,
        buffer=None,
        all_tokens=tr.trainer.llm_val_tokens,
        cfg=tr.trainer.cfg.data_cfg.model_cfg.acts_cfg,
        bos_processed_with_hook=False,
        num_batches=5,
    ),
)
with torch.inference_mode():
    cache = tr.trainer.get_cache()
    model.loss(next(tr.data), cache=cache)
    print(cache.logdict())
print(tr.trainable.training)  # buf = tr.trainer.get_databuffer(num_workers=0)
model.eval()
model.training
tr.trainable.train()
print(tr.trainer.train)

cache = tr.trainer.get_cache()
tr.trainable.loss(next(tr.data), cache=cache)
print(cache.logdict())
print(tr.trainable.training)  # buf = tr.trainer.get_databuffer(num_workers=0)

# tr.trainer.train(num_steps=2)
# tr.trainer.log_recons("recons/no_bos/", False)

# %%
tr.trainable.train()
tr.trainer.train()
# tr.trainer.post_step()

tokens = tr.cfg.train_cfg.data_cfg.get_split_tokens("test")
# %%
# tl_subj_model = tr.cfg.train_cfg.data_cfg.model_cfg.model
from nnsight import LanguageModel

# nnsm = LanguageModel(subj_model)
nnsm = LanguageModel("openai-community/gpt2", device_map="cuda")


def tokgen():
    i = 0
    while True:
        yield tokens[i * 256 : (i + 1) * 256]


tg = tokgen()

# %%
if False:
    x = next(tr.data)
    y = model(x)
    (y - x).pow(2).mean()
    cache = TrainCache()
    cache.L2_loss = ...
    model.loss(x, cache=cache)
    cache.L2_loss

# %%

# %%
from transformers import GPT2Model

# subj_model = GPT2Model.from_pretrained("gpt2")
# %%
tokens.shape
tb = tokens[:24]
if False:
    with nnsm.trace(tb):
        y = nnsm.output.save()

    nny = y.value.logits
    tly = tl_subj_model(tb)

    type(nny), type(tly)

    nls = torch.log_softmax(nny, dim=-1)
    tls = torch.log_softmax(tly, dim=-1)

    torch.allclose(nls, tls, atol=1e-2)
# %%
tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site
nnsight_name = "transformer.h.6.ln_1.input"

# %%


with nnsm.trace(tb):
    acts = acc(nnsm, nnsight_name).save()


acts_list = []
# %%


def hook_fn(acts, hook):
    acts_list.append(acts)


tl_subj_model.run_with_hooks(
    tb,
    stop_at_layer=tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.layer_num + 1,
    fwd_hooks=[(tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site, hook_fn)],
)

# %%
# %%
torch.allclose(acts_list[0], acts.value[0][0], rtol=1, atol=2)
# %%
acts_list[0] - acts.value[0][0]

# %%
acts_list[0] - acts_list[1]
# %%
nnsm.transformer.wte


# %%
import time


nnsight_name_0 = "transformer.h.1.ln_1.input"
nnsight_name_1 = "transformer.h.11.ln_1.input"


def test(name):
    t0 = time.time()
    with nnsm.trace(next(tg)):
        acts = acc(nnsm, name).save()
        acc(nnsm, "transformer.h.2.ln_1")

    print(acts.value)
    print(f"{name} took {time.time() - t0:.2f} seconds")


with torch.inference_mode():

    test(nnsight_name_0)
    test(nnsight_name_1)
    test(nnsight_name_0)
    test(nnsight_name_1)


# %%
with torch.inference_mode():

    with nnsm.trace(next(tg)):
        acts2 = acc(nnsm, "transformer.h.1.ln_1.input").save()
        acc(nnsm, "transformer.h.2.ln_1").input = "abc"
# %%
acts2.value


# %%
import time


def getname(layer):
    return f"transformer.h.{layer}.ln_1"


def test(layer, n=10):
    t0 = time.time()
    # print(t0)
    with torch.inference_mode():
        for i in range(n):
            try:
                with nnsm.trace(next(tg)):
                    zacts = acc(nnsm, getname(layer)).input.save()
                    # acc(nnsm, getname(layer + 1)).input = "abc"
            except Exception as e:
                # print(e)
                pass
            z = zacts.value[0][0] + 1
    print(f"{layer} took {time.time() - t0:.2f} seconds")
    return z


test(10)
test(10)
test(1)
test(10)
test(1)
test(1)

with torch.inference_mode():
    with nnsm.trace(next(tg)):
        zacts = acc(nnsm, getname(11)).input.save()

# %%
