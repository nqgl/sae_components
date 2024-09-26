from saeco.data import (
    ActsDataConfig,
    DataConfig,
    DataGenerationProcessConfig,
    ModelConfig,
    SplitConfig,
)


cfg = DataConfig()
model = cfg.model_cfg.model
print()
from transformers import GPT2Model
