# %%
from pathlib import Path
from saeco.components.sae_cache import SAECache
from saeco.trainer import Trainable
from saeco.architectures.threshgate_gradjust.tg_grad_deep_model import (
    cfg,
    deep_tg_grad_sae,
    DeepConfig,
)
from saeco.architectures.anth_update import anth_update_model, cfg as anth_cfg
from saeco.trainer.runner import RunConfig

# from saeco.architectures.threshgate import multigate_sae
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache

modelss = Path.home() / "workspace/saved_models/"


name = "(lars)multigate_sae0.001[25.0]-8273_1001"
name = "sweep_None/(lars)anth_update_model0.001[30.3]-95_10000"
# name = "binarize/(lars)deep_tg_grad_sae0.001[512.0]-40_100001"
name = "binarize/sweep_r36qclrk/(lars)deep_tg_grad_sae0.0003[512.0]-14_100001"


def load(cfg: BaseModel, model_fn, name, modify_cfg=lambda x: x, cfg_only=False):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = modify_cfg(cfg.model_validate_json(cfg_path.read_text()))
    tr = TrainingRunner(cfg, model_fn)
    if not cfg_only:
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


def modcfg(c: RunConfig[DeepConfig]):
    c.train_cfg.lr = 3e-4
    c.train_cfg.betas = (0.9, 0.995)
    c.train_cfg.coeffs = {
        k: v / old_loss_reduce_factor for k, v in c.train_cfg.coeffs.items()
    }
    for l in rm_losses:
        del c.train_cfg.coeffs[l]
    c.train_cfg.intermittent_metric_freq = 200
    # c.train_cfg.l0_target = 25
    c.train_cfg.raw_schedule_cfg.resample_period = 200_000
    # c.train_cfg.l0_targeting_enabled = True
    c.init_cfg.dict_mult = 1 / 4
    # c.arch_cfg.decay_l1_to = 1
    c.arch_cfg.decay_l1_end = 10_000
    c.train_cfg.checkpoint_period = 5_000
    return c


from saeco.data import ActsDataConfig


@torch.autocast(device_type="cuda", dtype=torch.float32)
def replacement_hook(
    acts,
    hook,
    encoder,
    cfg: ActsDataConfig,
    bos_processed_with_hook=False,
    saved_acts=False,
):
    sacts = acts
    if not bos_processed_with_hook:
        sacts = acts[:, 1:]

    acts_shape = sacts.shape
    acts_re = sacts.reshape(-1, cfg.d_data)
    if isinstance(saved_acts, list):
        saved_acts.append(acts_re)
    mlp_post_reconstr = encoder(acts_re)

    mlp_post_reconstr = mlp_post_reconstr.reshape(acts_shape)
    if bos_processed_with_hook:
        return mlp_post_reconstr
    return torch.cat((acts[:, :1], mlp_post_reconstr), dim=1)


from saeco.trainer.recons import get_recons_loss


def get_subst_loss(
    tokens,
    model,
    encoder,
    cfg: ActsDataConfig = None,
    bos_processed_with_hook=False,
    save_acts=False,
):
    model_prev_train_state = model.training
    model.eval()
    cfg = cfg or encoder.cfg
    if save_acts:
        save_acts = []
    with torch.autocast(device_type="cuda"):
        # loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[
                (
                    cfg.hook_site,
                    lambda *a, **k: replacement_hook(
                        *a,
                        **k,
                        encoder=encoder,
                        cfg=cfg,
                        bos_processed_with_hook=bos_processed_with_hook,
                        saved_acts=save_acts,
                    ),
                )
            ],
        )
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg.act_name, mean_ablate_hook)])
    model.train(model_prev_train_state)
    if save_acts:
        return recons_loss, save_acts[0]
    return recons_loss


import tqdm


# %%
# get_recons_loss(
#     tl_subj_model,
#     sae,
#     tg,
#     all_tokens=tokens,
#     cfg=tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg,
# )
# %%


from saeco.components import Loss

from saeco.trainer import TrainConfig


class CELossFromTokens(Loss):
    def __init__(self, model, cfg: TrainConfig):
        super().__init__(model)
        self.cfg = cfg

    def forward(self, x, *, y=None, cache: cl.Cache, **kwargs):
        sm = self.cfg.data_cfg.model_cfg.model
        assert y is None
        loss, subj_acts = get_subst_loss(
            x,
            sm,
            cache(self).module,
            self.cfg.data_cfg.model_cfg.acts_cfg,
            bos_processed_with_hook=not self.cfg.data_cfg.model_cfg.acts_cfg.excl_first,
            save_acts=True,
        )
        cache._ancestor.subj_acts = ...
        cache._ancestor.subj_acts = subj_acts
        return loss

    def loss(self, x, y, y_pred, cache: SAECache):
        assert False


class WrapLossGiveActs(Loss):
    def __init__(self, model: Loss):
        super().__init__(model)

    def forward(self, x, *, y=None, cache: cl.Cache, **kwargs):
        return self.module(cache._ancestor.subj_acts, cache=cache)

    def loss(self, x, y, y_pred, cache: SAECache):
        assert False


rm_losses = []  # ["L2_loss"]
old_loss_reduce_factor = 100

tr = load(cfg, deep_tg_grad_sae, name, modcfg, cfg_only=True)
tr = TrainingRunner(modcfg(cfg), deep_tg_grad_sae)
# tr = load(anth_cfg, anth_update_model, name)
sae = tr.trainable
tokens = tr.cfg.train_cfg.data_cfg.get_split_tokens("train")
tl_subj_model = tr.cfg.train_cfg.data_cfg.model_cfg.model
from nnsight import LanguageModel

# nnsm = LanguageModel(subj_model)
# nnsm = LanguageModel("openai-community/gpt2", device_map="cuda")
from transformer_lens import HookedTransformer

import torch.nn as nn

wrapped_old = {k: WrapLossGiveActs(v) for k, v in sae.losses.items()}
for l in rm_losses:
    del wrapped_old[l]
sae.losses = nn.ModuleDict(
    {"CE_Loss": CELossFromTokens(sae.model, tr.cfg.train_cfg), **wrapped_old}
)
sae.metrics = nn.ModuleDict({})
tl_subj_model.eval()
tr.trainer.log_freq = 1


def tokgen(b):
    i = 0
    while True:
        yield tokens[i * b : (i + 1) * b]
        i += 1


def train():
    tg = tokgen(16)
    tr.trainer.train(tg)


train()

# %%


# @torch.no_grad()
# def test(sae):
#     sae.eval()
#     loss = 0
#     for tokens in test_tokens:
#         loss += get_subst_loss(
#             tokens,
#             tl_subj_model,
#             sae,
#             tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg,
#         )
#     sae.train()
#     print(f"average loss: {loss.item()/100}")


# test(sae)
# # %%
# from torchlars import LARS

# optim = LARS(torch.optim.RAdam(sae.parameters(), lr=1e-4))
# # %%
# # %%
# prog = tqdm.trange(500)
# # %%
# # sae.model.model.module.encoder.lin.targeting.value = 0
# # %%
# tl_subj_model.eval()
# for i in prog:
#     optim.zero_grad()
#     loss = get_subst_loss(
#         next(tg),
#         tl_subj_model,
#         sae,
#         tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg,
#     )
#     loss.backward()
#     tr.trainer.post_backward()
#     optim.step()
#     print(loss.item())
#     tr.trainer.post_step()


# # %%
# if False:
#     x = next(tr.data)
#     y = model(x)
#     (y - x).pow(2).mean()
#     cache = TrainCache()
#     cache.L2_loss = ...
#     model.loss(x, cache=cache)
#     cache.L2_loss

# # %%

# # %%
# from transformers import GPT2Model

# # subj_model = GPT2Model.from_pretrained("gpt2")
# # %%
# tokens.shape
# tb = tokens[:24]
# if False:
#     with nnsm.trace(tb):
#         y = nnsm.output.save()

#     nny = y.value.logits
#     tly = tl_subj_model(tb)

#     type(nny), type(tly)

#     nls = torch.log_softmax(nny, dim=-1)
#     tls = torch.log_softmax(tly, dim=-1)

#     torch.allclose(nls, tls, atol=1e-2)
# # %%
# tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site
# nnsight_name = "transformer.h.6.ln_1.input"

# # %%


# with nnsm.trace(tb):
#     acts = acc(nnsm, nnsight_name).save()


# acts_list = []
# # %%


# def hook_fn(acts, hook):
#     acts_list.append(acts)


# tl_subj_model.run_with_hooks(
#     tb,
#     stop_at_layer=tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.layer_num + 1,
#     fwd_hooks=[(tr.cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site, hook_fn)],
# )

# # %%
# # %%
# torch.allclose(acts_list[0], acts.value[0][0], rtol=1, atol=2)
# # %%
# acts_list[0] - acts.value[0][0]

# # %%
# acts_list[0] - acts_list[1]
# # %%
# nnsm.transformer.wte


# # %%
# import time


# nnsight_name_0 = "transformer.h.1.ln_1.input"
# nnsight_name_1 = "transformer.h.11.ln_1.input"


# def test(name):
#     t0 = time.time()
#     with nnsm.trace(next(tg)):
#         acts = acc(nnsm, name).save()
#         acc(nnsm, "transformer.h.2.ln_1")

#     print(acts.value)
#     print(f"{name} took {time.time() - t0:.2f} seconds")


# with torch.inference_mode():

#     test(nnsight_name_0)
#     test(nnsight_name_1)
#     test(nnsight_name_0)
#     test(nnsight_name_1)


# # %%
# with torch.inference_mode():

#     with nnsm.trace(next(tg)):
#         acts2 = acc(nnsm, "transformer.h.1.ln_1.input").save()
#         acc(nnsm, "transformer.h.2.ln_1").input = "abc"
# # %%
# acts2.value


# # %%
# import time


# def getname(layer):
#     return f"transformer.h.{layer}.ln_1"


# def test(layer, n=10):
#     t0 = time.time()
#     # print(t0)
#     with torch.inference_mode():
#         for i in range(n):
#             try:
#                 with nnsm.trace(next(tg)):
#                     zacts = acc(nnsm, getname(layer)).input.save()
#                     # acc(nnsm, getname(layer + 1)).input = "abc"
#             except Exception as e:
#                 # print(e)
#                 pass
#             z = zacts.value[0][0] + 1
#     print(f"{layer} took {time.time() - t0:.2f} seconds")
#     return z


# test(10)
# test(10)
# test(1)
# test(10)
# test(1)
# test(1)

# with torch.inference_mode():
#     with nnsm.trace(next(tg)):
#         zacts = acc(nnsm, getname(11)).input.save()

# # %%
