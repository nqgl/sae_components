from functools import wraps

import nnsight
import saeco.core as cl
import torch
from context import model_name, storage_name

from jaxtyping import Float, Int
from pydantic import BaseModel

from rich.highlighter import Highlighter

from saeco.architectures.anth_update import anth_update_model, cfg
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.trainer import Trainable
from saeco.trainer.runner import TrainingRunner
from saeco.trainer.train_cache import TrainCache
from torch import Tensor


ec = Evaluation.from_model_name(model_name)
ec.store_acts(
    CachingConfig(
        dirname=storage_name,
        num_chunks=100,
        docs_per_chunk=300,
        documents_per_micro_batch=16,
        # exclude_bos_from_storage=True,
    ),
    displace_existing=True,
)
