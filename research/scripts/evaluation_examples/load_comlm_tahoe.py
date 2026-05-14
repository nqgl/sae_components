from comlm.exprank import XRNoisedBatch
from context import storage_name

from saeco.data.config._comlm_data_config_definitions import saeco_tahoe_data_cfg
from saeco.evaluation.evaluation import Evaluation
from saeco.mlog import mlog

mlog.init(project="markov-bio/evaluator")
root_eval = Evaluation[XRNoisedBatch].open_cache(storage_name)
root_eval.sae_cfg.train_cfg.data_cfg = saeco_tahoe_data_cfg  # type: ignore
