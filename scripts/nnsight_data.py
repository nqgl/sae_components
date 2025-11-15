from saeco.data import (
    DataConfig,
    DataGenerationProcessConfig,
    ModelConfig,
    SplitConfig,
)
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig


cfg = DataConfig()
model = cfg.model_cfg.model
print()
from transformers import GPT2Model
