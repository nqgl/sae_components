from collections.abc import Sequence
from saeco.data.config.model_config.hf_model_cfg import HuggingFaceModelConfig
from saeco.data.config.tokenization_config import (
    PackingMode,
    TokenizationConfig,
    TokenizationMode,
)
from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig

from .data_cfg import DataConfig, DataGenerationProcessConfig, SplitConfig
from .model_config.acts_data_cfg import ActsDataConfig
from .model_config.model_cfg import ModelConfig

gemma_2_2b_openwebtext = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        model_load_cfg=HuggingFaceModelConfig(
            model_name="google/gemma-2-2b",
        ),
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            sites=["model.layers.17.input"],
            storage_dtype_str="bfloat16",
            autocast_dtype_str=None,
        ),
        # model_name="google/gemma-2-2b",
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
        model_load_cfg=HuggingFaceModelConfig(
            model_name="google/gemma-2-2b",
        ),
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            sites=["model.layers.16.input"],
            storage_dtype_str="bfloat16",
            autocast_dtype_str=None,
        ),
        # model_name="google/gemma-2-2b",
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
        model_load_cfg=HuggingFaceModelConfig(
            model_name="google/gemma-2-2b",
        ),
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            sites=["model.layers.16.input"],
            storage_dtype_str="float32",
            autocast_dtype_str=None,
        ),
        # model_name="google/gemma-2-2b",
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
        model_load_cfg=HuggingFaceModelConfig(
            model_name="google/gemma-2-2b",
        ),
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2304,
            sites=["model.layers.16.input"],
            storage_dtype_str="float16",
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="float16",
        ),
        # model_name="google/gemma-2-2b",
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
            model_load_cfg=HuggingFaceModelConfig(
                model_name="google/gemma-2-2b",
            ),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=2304,
                sites=[f"model.layers.{layer}.input"],
                storage_dtype_str="float32",
                autocast_dtype_str=None,
            ),
            # model_name="google/gemma-2-2b",
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
    return DataConfig[HuggingFaceModelConfig](
        dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
        model_cfg=ModelConfig(
            model_load_cfg=HuggingFaceModelConfig(
                model_name="google/gemma-2-2b",
            ),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=2304,
                sites=[f"model.layers.{layer}.input"],
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
            ),
            # model_name="google/gemma-2-2b",
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
    cfg = DataConfig[HuggingFaceModelConfig](
        dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        model_cfg=ModelConfig[HuggingFaceModelConfig](
            model_load_cfg=HuggingFaceModelConfig(
                model_name="gpt2",
            ),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                sites=(
                    [f"transformer.h.{bp}" for bp in block_postfix]
                    if isinstance(block_postfix, list | tuple)
                    else [f"transformer.h.{block_postfix}"]
                ),
                d_data=768,
                autocast_dtype_str="bfloat16",
                force_cast_dtype_str="bfloat16",
                storage_dtype_str="bfloat16",
            ),
        ),
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=100_000_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**18,
            meta_batch_size=2**19,
            llm_batch_size=2**16,
        ),
        seq_len=256,
    )
    return cfg


def gpt_2_block(layer: int | list[int] | tuple[int], io="input"):
    if isinstance(layer, list | tuple):
        return gpt_2([f"{l}.{io}" for l in layer])
    return gpt_2(f"{layer}.{io}")


# ---------------------------------------------------------------------------
# Gemma 4 example configs
#
# Gemma 4 (released 2026-04-02, Apache 2.0) sizes: E2B/E4B (2/4B effective,
# multimodal/edge), 26B MoE, 31B dense. These are gated on HuggingFace: accept
# the license on the model page and export HF_TOKEN (or `huggingface-cli login`).
#
# `d_data` and layer-count values below target the E2B ("effective 2B") variant
# and should be reviewed once the exact architecture is loaded — they are
# placeholders intended to be overridden by the caller.
# ---------------------------------------------------------------------------

GEMMA_4_DEFAULT_MODEL = "google/gemma-4-E2B"
GEMMA_4_DEFAULT_MODEL_IT = "google/gemma-4-E2B-it"
GEMMA_4_DEFAULT_D_DATA = 1536


def gemma_4_openwebtext_bf16(
    layer: int = 5,
    model_name: str = GEMMA_4_DEFAULT_MODEL,
    d_data: int = GEMMA_4_DEFAULT_D_DATA,
):
    """Raw-text Gemma-4 config: tokenizes openwebtext on the fly with PACK."""
    return DataConfig[HuggingFaceModelConfig](
        dataset="Skylion007/openwebtext",
        model_cfg=ModelConfig(
            model_load_cfg=HuggingFaceModelConfig(model_name=model_name),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=d_data,
                sites=[f"model.layers.{layer}.input"],
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
            ),
            torch_dtype_str="bfloat16",
        ),
        trainsplit=SplitConfig(start=0, end=25, tokens_from_split=250_000_000),
        valsplit=SplitConfig(start=90, end=100, tokens_from_split=2_000_000),
        tokenization=TokenizationConfig(
            mode=TokenizationMode.RAW_TEXT,
            text_column="text",
            packing=PackingMode.PACK,
        ),
        # Raw-text packing inserts BOS via the tokenizer's add_special_tokens,
        # so we don't want `set_bos` forcing position 0 at read time.
        set_bos=False,
        generation_config=DataGenerationProcessConfig(
            acts_per_pile=2**15,
            meta_batch_size=2**18,
            llm_batch_size=2**16,
        ),
        seq_len=1024,
    )
    # len(model.model.language_model.layers)


class ModelDataSpec(SweepableConfig):
    model_name: str
    layers_name: str
    model_dim: int = 0
    mlp_expansion_factor: int = 0
    default_device: str = "cuda"

    def get_layer_names(self, *layers: int, suffixes: Sequence[str] = ("input",)):
        return [
            f"{self.layers_name}.{layer}.{suffix}"
            for suffix in suffixes
            for layer in layers
        ]

    def get_config_for_layers(self, *layers: int, suffixes: Sequence[str] = ("input",)):
        return ModelConfig(
            model_load_cfg=HuggingFaceModelConfig(
                model_name=self.model_name,
            ),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=self.model_dim,
                sites=self.get_layer_names(*layers, suffixes=suffixes),
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
                filter_pad=False,
            ),
            torch_dtype_str="bfloat16",
            device=self.default_device,
        )


gemma_4 = ModelDataSpec(
    model_name=GEMMA_4_DEFAULT_MODEL_IT, layers_name="model.language_model.layers"
)

gemma_4_it = ModelDataSpec(
    model_name=GEMMA_4_DEFAULT_MODEL, layers_name="model.language_model.layers"
)


def gemma_4_lmsys_chat(
    *layers: int,
    model_conf: ModelDataSpec = gemma_4_it,
    num_train_tokens: int = 50_000_000,
):
    """Gemma-4 applied to user conversation data (lmsys-chat-1m).

    Uses PAD mode so role/turn boundaries aren't scrambled across unrelated
    conversations. Features extracted this way see both user and model turns.
    """
    return DataConfig[HuggingFaceModelConfig](
        dataset="lmsys/lmsys-chat-1m",
        model_cfg=model_conf.get_config_for_layers(*layers),
        trainsplit=SplitConfig(start=0, end=90, tokens_from_split=num_train_tokens),
        valsplit=SplitConfig(start=90, end=100, tokens_from_split=1_000_000),
        tokenization=TokenizationConfig(
            mode=TokenizationMode.CONVERSATION,
            messages_column="conversation",
            packing=PackingMode.PAD,
            chat_template_kwargs={
                "add_generation_prompt": False,
                "enable_thinking": False,
            },
        ),
        set_bos=False,
        generation_config=DataGenerationProcessConfig(
            acts_per_pile=2**15,
            meta_batch_size=2**16,
            llm_batch_size=2**13,
        ),
        seq_len=1024,
    )


def gemma_4_reasoning_traces(
    layer: int = 5,
    model_name: str = GEMMA_4_DEFAULT_MODEL,
    d_data: int = GEMMA_4_DEFAULT_D_DATA,
    dataset: str = "lmsys/lmsys-chat-1m",
):
    """Gemma-4 conversation config with thinking enabled.

    Useful for extracting features from the model's own reasoning traces.
    `strip_historical_thinking=True` honors Gemma's rule that prior-turn
    assistant thoughts must not be re-fed in multi-turn chats.
    """
    return DataConfig[HuggingFaceModelConfig](
        dataset=dataset,
        model_cfg=ModelConfig(
            model_load_cfg=HuggingFaceModelConfig(model_name=model_name),
            acts_cfg=ActsDataConfig(
                excl_first=True,
                d_data=d_data,
                sites=[f"model.layers.{layer}.input"],
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
                filter_pad=False,
            ),
            torch_dtype_str="bfloat16",
        ),
        trainsplit=SplitConfig(start=0, end=90, tokens_from_split=25_000_000),
        valsplit=SplitConfig(start=90, end=100, tokens_from_split=1_000_000),
        tokenization=TokenizationConfig(
            mode=TokenizationMode.CONVERSATION,
            messages_column="conversation",
            packing=PackingMode.PAD,
            chat_template_kwargs={
                "add_generation_prompt": False,
                "enable_thinking": True,
            },
            strip_historical_thinking=True,
        ),
        set_bos=False,
        generation_config=DataGenerationProcessConfig(
            acts_per_pile=2**15,
            meta_batch_size=2**16,
            llm_batch_size=2**13,
        ),
        seq_len=2048,
    )
