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

data = data_cfg.train_data_batch_generator(model=model, batch_size=4096)


from sae_components.architectures.gated import gated_sae, test_train

model, losses = gated_sae(768, 8 * 768)

test_train(model, losses, data)
