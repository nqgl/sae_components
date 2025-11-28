from comlm.datasource.training_batch import NoisedBatch
from context import storage_name

from saeco.evaluation.evaluation import Evaluation
from saeco.mlog import mlog

mlog.init(project="markov-bio/evaluator")
root_eval = Evaluation[NoisedBatch].from_cache_name(storage_name)
print()
