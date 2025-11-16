from comlm.utils import (
    ComposerModelName,
)

from saeco.data.config.data_cfg import (
    DataConfig,
    DataGenerationProcessConfig,
    SplitConfig,
)
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig

model = "1761861357-rustling-mule"
# checkpoint = "ep0-ba198000-rank0.pt"
model = ComposerModelName.from_str("1762986288-acoustic-asp")
# model.get_latest_checkpoint_from_s3().fetch_from_s3()
data_cfg = DataConfig[ComlmModelConfig](
    override_dictpiler_path_str="/home/g/markov/sample_data_comlm",
    dataset="custom",
    model_cfg=ModelConfig[ComlmModelConfig](
        use_custom_data_source=False,
        model_load_cfg=ComlmModelConfig(
            chk_ident=model.get_latest_downloaded_checkpoint()
        ),
        acts_cfg=ActsDataConfig(
            filter_pad=False,
            excl_first=False,
            d_data=512,
            sites=[
                "model.layers.8.output.0"
            ],  # .0 unpacks the tuple of (output, kv cache)
            storage_dtype_str="float32",
            autocast_dtype_str=None,
        ),
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=25, tokens_from_split=30_000 * 512),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)
if __name__ == "__main__":
    data_cfg.store_split(data_cfg.trainsplit)
