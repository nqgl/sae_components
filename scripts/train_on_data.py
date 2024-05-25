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
data = data_cfg.train_data_batch_generator(model=model, batch_size=BATCH_SIZE)


from sae_components.architectures.gated import gated_sae
from sae_components.architectures.vanilla_tests import vanilla_sae

import wandb


def test_train(model, losses, data):
    from sae_components.trainer.trainer import Trainer, Trainable

    trainable = Trainable([model], losses).cuda()
    trainer = Trainer({}, trainable)
    wandb.init(project="sae-components", config={"model": repr(trainable)})

    trainer.train(data)


# model, losses = vanilla_sae(768, 8 * 768)
model, losses = gated_sae(768, 8 * 768)

test_train(model, losses, data)
