from .dataset import DataConfig, DataGenerationProcessConfig, SplitConfig
from .model_cfg import ActsDataConfig, ModelConfig

gemma_2_2b_openwebtext = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            site="model.layers.17.input",
        ),
        model_name="google/gemma-2-2b",
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=20, tokens_from_split=180_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**16,
        meta_batch_size=2**15,
        llm_batch_size=2**13,
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
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=300_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**18,
            meta_batch_size=2**18,
            llm_batch_size=2**16,
        ),
    )
