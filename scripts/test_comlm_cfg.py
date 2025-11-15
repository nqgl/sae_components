from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.data.config.data_cfg import (
    DataConfig,
    DataGenerationProcessConfig,
    SplitConfig,
)
from saeco.data.config.model_config.model_cfg import ModelConfig
from comlm.utils import (
    ModelCheckpointIdentifier,
    ComposerModelName,
    CheckpointSpecifier,
)

model = "1761861357-rustling-mule"
# checkpoint = "ep0-ba198000-rank0.pt"

data_cfg = DataConfig(
    dataset="jbloom/openwebtext_tokenized_gemma-2-9b",
    model_cfg=ModelConfig(
        use_custom_data_source=True,
        model_load_cfg=ComlmModelConfig(
            model_ident=ModelCheckpointIdentifier.from_model_name_and_batch(
                model_name=model, batch=198_000
            )
        ),
        acts_cfg=ActsDataConfig(
            excl_first=True,
            d_data=2048,
            sites=["model.layers.17.input"],
            storage_dtype_str="bfloat16",
            autocast_dtype_str=None,
        ),
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
