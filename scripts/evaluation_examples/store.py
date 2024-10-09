import nnsight
import saeco.core as cl
import torch
from context import model_name, storage_name

from saeco.architectures.anth_update import anth_update_model, cfg
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.trainer import Trainable
from saeco.trainer.runner import TrainingRunner
from saeco.trainer.train_cache import TrainCache
from torch import Tensor


root_eval = Evaluation.from_model_name(model_name)
root_eval.store_acts(
    CachingConfig(
        dirname=storage_name,
        num_chunks=100,
        docs_per_chunk=100,
        documents_per_micro_batch=32,
        # exclude_bos_from_storage=True,
        eager_sparse_generation=True,
        store_feature_tensors=False,
        deferred_blocked_store_feats_block_size=False,
        # metadatas_from_src_column_names=["tissue", "cell_type"],
    ),
    displace_existing=True,
)
