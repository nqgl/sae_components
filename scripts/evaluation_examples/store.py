from comlm.datasource.training_batch import NoisedBatch
from context import model_name, storage_name

from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.storage.saved_acts_config import CachingConfig
from saeco.mlog import mlog

mlog.init(project="nqgl/default-project")
root_eval = Evaluation.from_model_path(model_name)
root_eval.store_acts(
    CachingConfig[NoisedBatch](
        dirname=storage_name,
        num_chunks=100,
        docs_per_chunk=100,
        documents_per_micro_batch=32,
        # exclude_bos_from_storage=True,
        eager_sparse_generation=True,
        store_feature_tensors=False,
        deferred_blocked_store_feats_block_size=10,
        # metadatas_from_src_column_names=["tissue", "cell_type"],
    ),
    displace_existing=True,
)
