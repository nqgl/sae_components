from comlm.datasource.data_config_definitions import tahoe_data_config
from comlm.storage import ComposerModelName

from saeco.data.config.data_cfg import DataConfig
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.data.config.split_config import SplitConfig

saeco_tahoe_data_cfg = DataConfig[ComlmModelConfig](
    override_dictpiler_path_str="/home/g/workspace/tahoe_batches",
    dataset="tahoe_bulked",
    model_cfg=ModelConfig[ComlmModelConfig](
        model_load_cfg=ComlmModelConfig(
            chk_ident=ComposerModelName.from_str(
                "1762986288-acoustic-asp"
            ).get_latest_downloaded_checkpoint(),
            inject_arch_data_cfg=tahoe_data_config,
        ),
        acts_cfg=ActsDataConfig(
            filter_pad=False,
            excl_first=False,
            d_data=512,
            sites=["layers.6.output.0"],  # .0 unpacks the tuple of (output, kv cache)
            storage_dtype_str="float32",
            autocast_dtype_str=None,
        ),
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=100, tokens_from_split=None),
    generation_config=DataGenerationProcessConfig(
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)
