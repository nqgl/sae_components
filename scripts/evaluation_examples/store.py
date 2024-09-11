from context import model_name, storage_name
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.trainer import Trainable

from saeco.architectures.anth_update import cfg, anth_update_model

from jaxtyping import Int, Float
from torch import Tensor
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
from functools import wraps
from saeco.evaluation.evaluation import Evaluation
import nnsight

from rich.highlighter import Highlighter


ec = Evaluation.from_model_name(model_name)
ec.store_acts(
    CachingConfig(
        dirname=storage_name,
        num_chunks=70,
        docs_per_chunk=100,
        documents_per_micro_batch=16,
    ),
    displace_existing=True,
)
