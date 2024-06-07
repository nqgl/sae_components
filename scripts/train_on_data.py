# %%

from sae_components.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from sae_components.trainer.trainable import Trainable

gpt2 = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096


from sae_components.architectures.gated import gated_sae, gated_sae_no_detach
from sae_components.architectures.vanilla_tests import (
    vanilla_sae,
    basic_vanilla_sae,
    basic_vanilla_sae_lin,
    basic_vanilla_sae_lin_no_orth,
)
from sae_components.architectures.deep.deep import deep_sae, resid_deep_sae
from sae_components.architectures.deep.deep_resid_gated import (
    deep_resid_gated,
    deep_resid_gated2,
    deep_resid_gated2_wider,
    deep_resid_gated2_deeper,
    deep_resid_gated2_deeper_still,
    deep_resid_gated2_wider,
    deep_resid_gated2_wider2,
)
from sae_components.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb

import torch
import sys

sys.setrecursionlimit(10**5)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True


PROJECT = "nn.Linear Check"


def test_train(models, losses, name, l0_target=45, lr=3e-4):
    from sae_components.trainer.trainer import Trainer, TrainConfig

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 7e-4,
        },
        lr=lr,
        use_autocast=True,
        batch_size=4096,
        wandb_cfg=dict(project=PROJECT),
        betas=(0.9, 0.99),
    )
    trainable = Trainable(models, losses).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")

    trainer.train()


# models = [resid_deep_sae, deep_resid_gated2, gated_sae]
# models = [deep_catseq_resid, vanilla_sae, deep_catseq]
models = [deep_resid_gated2_deeper]
models = [deep_resid_gated2_deeper_still]
models = [deep_resid_gated2_wider2, deep_resid_gated2_wider]
models = [vanilla_sae]
models = [deep_sae]
models = [gated_sae_no_detach, gated_sae]
models = [basic_vanilla_sae_lin, basic_vanilla_sae_lin_no_orth]

# test_train(
#     *deep_resid_gated2(768, 768 * 8),
#     name="deep_resid_gated2",
#     l0_target=45,
#     lr=1e-3,
# )
# test_train(
#     *basic_vanilla_sae_lin(768, 768 * 32),
#     name="vanilla 32dm",
#     l0_target=45,
#     lr=1e-3,
# )
for model_fn in models:
    model, losses = model_fn(768, 8 * 768)

    test_train(model, losses, name=model_fn.__name__, l0_target=45, lr=1e-3)
    del (model, losses)

for model_fn in models:
    model, losses = model_fn(768, 8 * 768)

    test_train(model, losses, name=model_fn.__name__, l0_target=45)
    del (model, losses)
test_train(
    *deep_resid_gated2(768, 768 * 8, extra_layers=2, hidden_mult=4, mlp_mult=6),
    name="wider custom",
    l0_target=45,
)

for model_fn in models:
    model, losses = model_fn(768, 8 * 768)

    test_train(model, losses, name=model_fn.__name__, l0_target=15)
    del (model, losses)

# # %%
# model, losses = gated_sae(768, 8 * 768)


# # model.encoder.magnitude.weight.module.weight


# # %%
# model.decoder.weight.module.right.data == losses[
#     "L2_aux_loss"
# ].module.module.decoder.weight.right.data
# # %%
# model.decoder.weight.module.right.data
# # %%
# losses[
#     "L2_aux_loss"
# ].module.module.decoder.weight.right.requires_grad  # %% they are not synced up rn gotta fix that
# losses["L2_loss"].module.module.decoder.weight.right.requires_grad


# # %%
