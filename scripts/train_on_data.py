# %%

from saeco.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

from saeco.trainer.trainable import Trainable

gpt2 = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096

from saeco.architectures.initialization.geo_med import getmed, getmean
from saeco.architectures.gated import gated_sae, gated_sae_no_detach
from saeco.architectures.gate_hierarch import gate_two_weights
from saeco.architectures.vanilla_tests import (
    vanilla_sae,
    basic_vanilla_sae,
    basic_vanilla_sae_lin,
    basic_vanilla_sae_lin_no_orth,
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
from saeco.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb
from saeco.architectures.remax import (
    remax_sae,
    remax1_sae,
    remaxk_sae,
    remaxkv_sae,
    remaxkB_sae,
    remaxkvB_sae,
)
from saeco.architectures.topk import topk_sae
import torch
import sys
from saeco.architectures.initialization.initializer import Initializer
from saeco.trainer.normalizers import (
    ConstL2Normalizer,
    Normalized,
    Normalizer,
    L2Normalizer,
)


torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True


PROJECT = "nn.Linear Check"


# def test_train(models, losses, name, l0_target=45, lr=3e-4):
#     from saeco.trainer.trainer import Trainer, TrainConfig

#     cfg = TrainConfig(
#         l0_target=l0_target,
#         coeffs={
#             "sparsity_loss": 2e-3 if l0_target is None else 7e-4,
#         },
#         lr=lr,
#         use_autocast=True,
#         batch_size=4096,
#         wandb_cfg=dict(project=PROJECT),
#         betas=(0.9, 0.99),
#     )
#     trainable = Trainable(models, losses).cuda()
#     trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")


#     trainer.train()


def test_train(model_fn, l0_target=45, lr=3e-4, normalizer=None, **kwargs):
    from saeco.trainer.trainer import Trainer, TrainConfig

    normalizer = normalizer or ConstL2Normalizer()

    name = model_fn.__name__

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 14e-4,
        },
        lr=lr,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
        batch_size=4096,
        use_lars=True,
        kwargs=kwargs,
    )
    buf = iter(cfg.data_cfg.get_databuffer())
    normalizer.prime_normalizer(buf)
    models, losses = model_fn(
        Initializer(
            768,
            dict_mult=8,
            l0_target=l0_target,
            median=getmed(buf=buf, normalizer=normalizer),
        ),
        **kwargs,
    )

    trainable = Trainable(models, losses, normalizer=normalizer).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name + f"_{lr:.0e}")
    trainable.normalizer.prime_normalizer(buf)
    trainer.post_step()
    trainer.train(buf)


# test_train(gated_sae, l0_target=45)
test_train(topk_sae, l0_target=45)
test_train(gate_two_weights, lr=1e-3, l0_target=45)

# for i in range(20, 85, 5):
#     test_train(remaxk_sae, lr=1e-3, l0_target=45, basescale=i)
# for i in range(20, 85, 5):
#     test_train(remaxk_sae, l0_target=45, basescale=i)

# models = [resid_deep_sae, deep_resid_gated2, gated_sae]

# models = [deep_catseq_resid, vanilla_sae, deep_catseq]
models = [deep_resid_gated2_deeper]
models = [deep_resid_gated2_deeper_still]
models = [deep_resid_gated2_wider2, deep_resid_gated2_wider]
models = [vanilla_sae]
models = [deep_sae]
models = [gated_sae_no_detach, gated_sae]
models = [
    remaxk_sae,
    remax_sae,
    remaxk_sae,
    remaxkB_sae,
    remaxkv_sae,
    remaxkvB_sae,
]
models = [gate_two_weights]


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
    test_train(model_fn, l0_target=20, lr=1e-3)
    del model_fn

for model_fn in models:

    test_train(model_fn, l0_target=45)
    del model_fn
test_train(
    *deep_resid_gated2(768, 768 * 8, extra_layers=2, hidden_mult=4, mlp_mult=6),
    name="wider custom",
    l0_target=45,
)

for model_fn in models:
    test_train(model_fn, l0_target=15)
    del model_fn

# # %%
# model_fn = gated_sae(768, 8 * 768)


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
