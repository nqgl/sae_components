# %%
from pathlib import Path

import torch
from pydantic import BaseModel
from saeco.architectures.dg_test.dg_gate import cfg, gated_dg_sae

from saeco.trainer.runner import TrainingRunner
from saeco.trainer.train_cache import TrainCache

modelss = Path.home() / "workspace/saved_models/"

name = "(lars)gated_dg_sae0.003[25.0]-8275_29981"


def load(cfg: BaseModel, model_fn, name):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    tr = TrainingRunner(cfg, model_fn)
    tr.trainable.load_state_dict(torch.load(pt_path))
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


tr = load(cfg, gated_dg_sae, name)
model = tr.trainable
tokens = tr.cfg.train_cfg.data_cfg.get_split_tokens("test")
tl_subj_model = tr.cfg.train_cfg.data_cfg.model_cfg.model
from nnsight import LanguageModel

# nnsm = LanguageModel(subj_model)
nnsm = LanguageModel("openai-community/gpt2", device_map="cuda")


def tokgen():
    i = 0
    while True:
        yield tokens[i * 256 : (i + 1) * 256]


tg = tokgen()
# %%
x = next(tr.data)
acts = model.get_acts(x)
acts
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
            except Exception:
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
