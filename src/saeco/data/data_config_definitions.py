from saeco.sweeps.sweepable_config import Swept
from .data_cfg import DataConfig, DataGenerationProcessConfig, SplitConfig
from .model_cfg import ActsDataConfig, ModelConfig


gemma_2_2b_openwebtext = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            site="model.layers.17.input",
            storage_dtype_str="bfloat16",
            autocast_dtype_str=None,
        ),
        model_name="google/gemma-2-2b",
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=25, tokens_from_split=250_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)

gemma_2_2b_openwebtext_test = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            site="model.layers.16.input",
            storage_dtype_str="bfloat16",
            autocast_dtype_str=None,
        ),
        model_name="google/gemma-2-2b",
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=25, tokens_from_split=10_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)
gemma_2_2b_openwebtext_test_fp32 = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            site="model.layers.16.input",
            storage_dtype_str="float32",
            autocast_dtype_str=None,
        ),
        model_name="google/gemma-2-2b",
        torch_dtype_str=None,
    ),
    trainsplit=SplitConfig(start=0, end=23, tokens_from_split=10_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**17,
        meta_batch_size=2**16,
        llm_batch_size=2**12,
    ),
    seq_len=1024,
)

gemma_2_2b_openwebtext_test_fp16 = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            site="model.layers.16.input",
            storage_dtype_str="float16",
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="float16",
        ),
        model_name="google/gemma-2-2b",
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=24, tokens_from_split=10_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)
# gemma_2_2b_openwebtext_fp32 = DataConfig(
#     dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
#     model_cfg=ModelConfig(
#         acts_cfg=ActsDataConfig(
#             excl_first=True,
#             d_data=2304,
#             site="model.layers.17.input",
#             storage_dtype_str="float32",
#             autocast_dtype_str=None,
#         ),
#         model_name="google/gemma-2-2b",
#         torch_dtype_str=None,
#     ),
#     trainsplit=SplitConfig(start=0, end=25, tokens_from_split=25_000_000),
#     generation_config=DataGenerationProcessConfig(
#         # tokens_per_pile=2**25,
#         acts_per_pile=2**15,
#         meta_batch_size=2**17,
#         llm_batch_size=2**14,
#     ),
#     seq_len=1024,
# )


def gemma_2_2b_openwebtext_fp32(layer=17):
    return DataConfig(
        dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=2304,
                site=f"model.layers.{layer}.input",
                storage_dtype_str="float32",
                autocast_dtype_str=None,
            ),
            model_name="google/gemma-2-2b",
            torch_dtype_str=None,
        ),
        trainsplit=SplitConfig(start=0, end=25, tokens_from_split=200_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**13,
            meta_batch_size=2**17,
            llm_batch_size=2**14,
        ),
        seq_len=1024,
    )


def gemma_2_2b_openwebtext_bf16(layer=17):
    return DataConfig(
        dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=2304,
                site=f"model.layers.{layer}.input",
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
            ),
            model_name="google/gemma-2-2b",
            torch_dtype_str="bfloat16",
        ),
        trainsplit=SplitConfig(start=0, end=25, tokens_from_split=250_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**15,
            meta_batch_size=2**18,
            llm_batch_size=2**16,
        ),
        seq_len=1024,
    )


def gpt_2(block_postfix):
    return DataConfig(
        dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True,
                site=(
                    Swept[str](
                        *[f"transformer.h.{bp}" for bp in block_postfix],
                    )
                    if isinstance(block_postfix, list | tuple)
                    else f"transformer.h.{block_postfix}"
                ),
                d_data=768,
                autocast_dtype_str="bfloat16",
                force_cast_dtype_str="bfloat16",
                storage_dtype_str="bfloat16",
            ),
            model_name="gpt2",
        ),
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=200_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**17,
            meta_batch_size=2**19,
            llm_batch_size=2**16,
        ),
        seq_len=256,
    )


def gpt_2_block(layer=7, io="input"):
    if isinstance(layer, list | tuple):
        return gpt_2([f"{l}.{io}" for l in layer])
    return gpt_2(f"{layer}.{io}")
