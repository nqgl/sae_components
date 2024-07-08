# %%


from saeco.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from saeco.trainer.trainable import Trainable

gpt2 = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096


from saeco.architectures.gated import gated_sae, gated_sae_no_detach
from saeco.architectures.vanilla_tests import (
    basic_vanilla_sae_lin,
)
from saeco.architectures.deep.deep import deep_sae, resid_deep_sae
from saeco.architectures.deep.deep_resid_gated import (
    deep_resid_gated,
    deep_resid_gated2,
    deep_resid_gated2_wider,
    deep_resid_gated2_deeper,
    deep_resid_gated2_deeper_still,
    deep_resid_gated2_wider,
    deep_resid_gated2_wider2,
)
from saeco.architectures.topk import topk_sae
from saeco.architectures.remax import remax_sae, remax1_sae
from saeco.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb
import torch
from saeco.architectures.initialization.initializer import Initializer

# def test_train(models, losses, name, l0_target=45, lr=3e-4):
#     from saeco.trainer.trainer import Trainer, TrainConfig

#     cfg = TrainConfig(
#         l0_target=l0_target,
#         coeffs={
#             "sparsity_loss": 2e-3 if l0_target is None else 7e-4,
#         },
#         lr=lr,
#     )
#     trainable = Trainable(models, losses).cuda()
#     trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")

#     trainer.train()


# # models = [resid_deep_sae, deep_resid_gated2, gated_sae]
# # models = [deep_catseq_resid, vanilla_sae, deep_catseq]
# models = [gated_sae_no_detach, gated_sae]
# models = [deep_resid_gated2_deeper]
# models = [deep_resid_gated2_deeper_still]
# models = [deep_resid_gated2_wider2, deep_resid_gated2_wider]
# models = [resid_deep_sae]
# # test_train(
# #     *deep_resid_gated2(768, 768 * 8),
# #     name="deep_resid_gated2",
# #     l0_target=45,
# #     lr=1e-3,
# # )
# for model_fn in models:
#     model, losses = model_fn(768, 8 * 768)

#     test_train(model, losses, name=model_fn.__name__, l0_target=45, lr=1e-3)
#     del (model, losses)
# test_train(
#     *deep_resid_gated2(768, 768 * 8, extra_layers=2, hidden_mult=4, mlp_mult=6),
#     name="wider custom",
#     l0_target=45,
#     lr=1e-3,
# )

# for model_fn in models:
#     model, losses = model_fn(768, 8 * 768)

#     test_train(model, losses, name=model_fn.__name__, l0_target=45)
#     del (model, losses)
# test_train(
#     *deep_resid_gated2(768, 768 * 8, extra_layers=2, hidden_mult=4, mlp_mult=6),
#     name="wider custom",
#     l0_target=45,
# )

# for model_fn in models:
#     model, losses = model_fn(768, 8 * 768)

#     test_train(model, losses, name=model_fn.__name__, l0_target=15)
#     del (model, losses)

# # # %%
# # model, losses = gated_sae(768, 8 * 768)


# # # model.encoder.magnitude.weight.module.weight


# # # %%
# # model.decoder.weight.module.right.data == losses[
# #     "L2_aux_loss"
# # ].module.module.decoder.weight.right.data
# # # %%
# # model.decoder.weight.module.right.data
# # # %%
# # losses[
# #     "L2_aux_loss"
# # ].module.module.decoder.weight.right.requires_grad  # %% they are not synced up rn gotta fix that
# # losses["L2_loss"].module.module.decoder.weight.right.requires_grad


# # # %%

# # %%

# # models = model

# # model = models[0]
# # %%

# trained_model = model[0]
# # %%
# import torch


def cmpmodel(model):
    mats = [
        torch.eye(25).cuda(),
        model.encoder.project_up.right,
        model.encoder.layers[0][0].weight.right,
        model.encoder.layers[1][0].weight.right,
        torch.nn.init.kaiming_uniform_(torch.empty(768, 768), mode="fan_in").cuda(),
        torch.nn.init.kaiming_uniform_(torch.empty(768, 8 * 768)).cuda(),
    ]

    for mat in mats:
        i, o = mat.shape
        vecs_i = torch.randn(64, i).cuda()
        vni = vecs_i / vecs_i.norm(dim=1, keepdim=True)
        vecs_o = torch.randn(64, o).cuda()
        vno = vecs_o / vecs_o.norm(dim=1, keepdim=True)
        mvi = vni @ mat
        mvo = vno @ mat.T
        in_mag = mvi.norm(dim=1).mean()
        out_mag = mvo.norm(dim=1).mean()
        print(in_mag, out_mag)


# %%


# model, losses = resid_deep_sae(768, 8 * 768)
# new_model = model[0]
# cmpmodel(new_model.cuda())
# %%

# cmpmodel(trained_model)
# %%
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from jaxtyping import Float

from saeco.architectures.initialization.tools import bias
from saeco.architectures.initialization.tools import weight
from saeco.architectures.initialization.tools import mlp_layer
from saeco.components.ops.detach import Thresh
import saeco.core as cl
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    Penalty,
    L1Penalty,
    FreqTracked,
    EMAFreqTracker,
    FreqTracker,
    Loss,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.core.basic_ops import Add, MatMul, Sub, Mul
from typing import Optional
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft
import saeco.components as co


from saeco.core.collections.seq import ResidualSeq
from saeco.trainer.trainable import Trainable

# %%
d_data = 768
d_dict = d_data * 8


def deep(
    extra_layers=4,
    hidden_mult=1,
    mlp_mult=2,
    layer_nonlinearity=nn.LeakyReLU,
    scale=1.0,
):
    d_hidden = d_data * hidden_mult

    layers = Seq(
        *[
            Seq(
                mlp_layer(d_hidden, d_hidden * mlp_mult, scale=scale),
                # nn.LayerNorm(d_hidden, device="cuda"),
            )
            for i in range(extra_layers)
        ],
    )
    return layers


d = deep(scale=2 ** (-0.25)).cuda()


def test(model):
    vecs = torch.randn(64, d_data).cuda()
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    m = model(vecs, cache=SAECache())
    return m.norm(dim=1).mean()


test(d)


# %%
@torch.no_grad()
def perturb_element(p: Tensor):
    adj = 1
    pi = p
    for i in range(len(p.shape)):
        ni = torch.randint(0, p.shape[i] - 1, (1,)).item()
        pi = pi[ni]
        print(pi.shape)
    pi += adj

    def undo():
        nonlocal pi
        pi -= adj

    return undo


@torch.no_grad()
def perturb_p(p: Tensor):
    adj = torch.randn_like(p) * 0.01
    p[:] = p + adj

    def undo():
        nonlocal p
        p[:] = p - adj

    return undo


@torch.no_grad()
def sensitivity(model, p, perturb_fn=perturb_element):
    vecs = torch.randn(64, d_data).cuda()
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    out = model(vecs, cache=SAECache())

    undo = perturb_fn(p)
    out_p = model(vecs, cache=SAECache())
    undo()
    return (out_p - out).norm(dim=-1).mean()


def sensitivity_batched(model, p, perturb_fn=perturb_element, batches=10):
    vals = [sensitivity(model, p, perturb_fn) for _ in range(batches)]
    return torch.tensor(vals).mean()


for p in d.parameters():
    print([sensitivity(d, p, perturb_p).item() for _ in range(10)])


# %%
opt = torch.optim.Adam(d.parameters(), lr=1e-3)


# %%


def conditioned_groups(model, d_data, lr=1e-3, perturb_fn=perturb_element):
    @torch.no_grad()
    def sensitivity(model, p, perturb_fn=perturb_element):
        vecs = torch.randn(64, d_data).cuda()
        vecs = vecs / vecs.norm(dim=1, keepdim=True)
        out = model(vecs, cache=SAECache())

        undo = perturb_fn(p)
        out_p = model(vecs, cache=SAECache())
        undo()
        return (out_p - out).norm(dim=-1).mean()

    def sensitivity_batched(model, p, perturb_fn=perturb_element, batches=10):
        vals = [sensitivity(model, p, perturb_fn) for _ in range(batches)]
        return torch.tensor(vals).mean()

    sens = [(sensitivity_batched(model, p, perturb_fn), p) for p in model.parameters()]
    cmax = max([s[0] for s in sens])
    groups = [{"params": [s[1]], "lr": lr / (s[0] / cmax).item()} for s in sens]
    return groups


# %%


@torch.no_grad()
def perturb_element(p: Tensor):
    adj = 1
    pi = p
    for i in range(len(p.shape)):
        ni = torch.randint(0, p.shape[i] - 1, (1,)).item()
        pi = pi[ni]
        print(pi.shape)
    pi += adj

    def undo():
        nonlocal pi
        pi -= adj

    return undo


@torch.no_grad()
def perturb_p(p: Tensor):
    adj = torch.randn_like(p) * 0.01
    p[:] = p + adj

    def undo():
        nonlocal p
        p[:] = p - adj

    return undo


@torch.no_grad()
def conditioned_groups(model, d_data, lr=1e-3, perturb_fn=perturb_element):
    def sensitivity(model, p, perturb_fn=perturb_element):
        vecs = torch.randn(64, d_data).cuda()
        vecs = vecs / vecs.norm(dim=1, keepdim=True)
        out = model(vecs, cache=SAECache())

        undo = perturb_fn(p)
        out_p = model(vecs, cache=SAECache())
        undo()
        return (out_p - out).norm(dim=-1).mean()

    def sensitivity_batched(model, p, perturb_fn=perturb_element, batches=10):
        vals = [sensitivity(model, p, perturb_fn) for _ in range(batches)]
        v = torch.tensor(vals)
        print(v.max(), v.mean(), v.min())
        return v.mean()

    sens = [(sensitivity_batched(model, p, perturb_fn), p) for p in model.parameters()]
    cmax = max([s[0] for s in sens])
    cmin = min([s[0] for s in sens])
    print("max,min", cmax, cmin)
    for s in sens:
        print(s[0], s[1].shape)

    groups = [{"params": [s[1]], "lr": lr / (s[0] / cmin).item()} for s in sens]

    return groups


# test_train(
#     *,
#     name="wider custom",
#     l0_target=45,
#     lr=1e-3,
# )
from saeco.trainer.normalizers import (
    ConstL2Normalizer,
    Normalized,
    Normalizer,
    L2Normalizer,
)


def train_cond(model_fn, l0_target=45, lr=3e-4):
    from saeco.trainer.trainer import Trainer, TrainConfig

    name = "(cond)" + model_fn.__name__
    models, losses = model_fn(768, 768 * 8)

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 7e-4,
        },
        lr=lr,
    )
    trainable = Trainable(models, losses).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")
    buf = trainer.get_databuffer()
    trainable.normalizer.prime_normalizer(buf)
    trainer.post_step()
    optim = torch.optim.RAdam(conditioned_groups(trainable, 768, lr=lr), lr=lr)
    trainer.optim = optim
    trainer.train(buf)


from torchlars import LARS


# %%
PROJECT = "nn.Linear Check"


def train_lars(model_fn, l0_target=45, lr=3e-4):
    from saeco.trainer.trainer import Trainer, TrainConfig

    name = "(lars)" + model_fn.__name__
    models, losses = model_fn(Initializer(768, dict_mult=8, l0_target=l0_target))

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 14e-4,
        },
        lr=lr,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
    )
    trainable = Trainable(models, losses, normalizer=L2Normalizer()).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")
    buf = iter(trainer.get_databuffer())
    trainable.normalizer.prime_normalizer(buf)
    trainer.post_step()
    optim = torch.optim.RAdam(trainable.parameters(), lr=lr, betas=(0.9, 0.99))
    trainer.optim = LARS(optim)
    trainer.train(buf)
