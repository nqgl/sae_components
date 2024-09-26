from .dataset import DataConfig, DataGenerationProcessConfig, SplitConfig
from .model_cfg import ActsDataConfig, ModelConfig

gemma_2_2b_openwebtext = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(excl_first=True, d_data=2048),
        model_name="gemma-2b",
        torch_dtype="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=50, tokens_from_split=500_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**16,
        meta_batch_size=2**17,
        llm_batch_size=2**15,
    ),
)


def gpt_2(layer=6, io="input"):
    return DataConfig(
        dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True, site=f"transformer.h.{layer}.{io}", d_data=768
            ),
            model_name="gpt2",
        ),
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=100_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**16,
            meta_batch_size=2**17,
            llm_batch_size=2**15,
        ),
    )
