# %%


from sae_components.data.sc.dataset import DataConfig, SplitConfig
from transformer_lens import HookedTransformer

data_cfg = DataConfig(
    trainsplit=SplitConfig(
        splitname="train",
        start=0,
        end=40,
        tokens_from_split=400_000_000 // 1,
    )
)

model = HookedTransformer.from_pretrained("gpt2")
BATCH_SIZE = 4096


from sae_components.architectures.gated import gated_sae
from sae_components.architectures.vanilla_tests import vanilla_sae
from sae_components.architectures.deep.deep import deep_sae, resid_deep_sae
from sae_components.architectures.deep.deep_resid_gated import (
    deep_resid_gated,
    deep_resid_gated2,
)
from sae_components.architectures.deep.catseq import deep_catseq, deep_catseq_resid
import wandb


def test_train(models, losses, data, name, l0_target=45):
    from sae_components.trainer.trainer import Trainer, Trainable, TrainConfig

    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 1e-4,
        },
    )
    trainable = Trainable(models, losses).cuda()
    trainer = Trainer(cfg, trainable, namestuff=name)

    trainer.train(data)


models = [resid_deep_sae, deep_resid_gated2, gated_sae]
# models = [deep_catseq_resid, vanilla_sae, deep_catseq]
for model_fn in models:
    data = data_cfg.train_data_batch_generator(model=model, batch_size=BATCH_SIZE)
    model, losses = model_fn(768, 8 * 768)

    test_train(model, losses, data, name=model_fn.__name__)
    del model, losses, data
for model_fn in models:
    data = data_cfg.train_data_batch_generator(model=model, batch_size=BATCH_SIZE)
    model, losses = model_fn(768, 8 * 768)

    test_train(model, losses, data, name=model_fn.__name__, l0_target=15)
    del model, losses, data

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
