from comlm.datasource.data_config_definitions import tahoe_data_config
from comlm.datasource.training_batch import NoisedBatch
from comlm.exprank import XRNoisedBatch
from comlm.storage import ComposerModelName
from context import model_name, storage_name

from saeco.data.config.data_cfg import DataConfig
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.data.config.split_config import SplitConfig
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.storage.cache_config import CacheConfig
from saeco.mlog import mlog
from saeco.data.config._comlm_data_config_definitions import convert_to_tahoe

data_cfg = DataConfig[ComlmModelConfig](
    override_token_dictpiler_path_str="/home/g/workspace/tahoe_batches",
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
            d_data=768,
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
    seq_len=2048,
)
# tahoe_data_config.single_cell_data.tokenized_piled_data.r_piler.num_samples
mlog.init(project="markov-bio/evaluator")
# data_cfg.model_cfg.model_load_cfg.pretrained_arch.run_cfg.train_cfg.data_cfg = (
#     tahoe_data_config
# )
# data_cfg.store_split(data_cfg.trainsplit)

root_eval = Evaluation[XRNoisedBatch].open_from_model(model_name)

root_eval.sae_cfg.train_cfg.data_cfg = convert_to_tahoe(
    root_eval.sae_cfg.train_cfg.data_cfg  # type: ignore
)
# root_eval.
# Path.home() / "workspace" / "tahoe_batches"
root_eval.store_acts(
    CacheConfig[XRNoisedBatch](
        dirname="tahoe_2048_x5_v4",
        num_chunks=423,  # 3,
        docs_per_chunk=512,
        documents_per_micro_batch=128,
        # exclude_bos_from_storage=True,
        eager_sparse_generation=True,
        store_feature_tensors=False,
        deferred_blocked_store_feats_block_size=None,
        src_piler_num_epochs=4,
        # metadatas_from_src_column_names=["tissue", "cell_type"],
    ),
    displace_existing=True,
)
