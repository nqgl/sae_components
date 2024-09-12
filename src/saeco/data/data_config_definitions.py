from .dataset import DataConfig, SplitConfig
from .model_cfg import ModelConfig, ActsDataConfig

gemma_2_2b_openwebtext = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(excl_first=True, d_data=2048),
        model_name="gemma-2b",
        torch_dtype="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=50, tokens_from_split=100_000_000),
)
