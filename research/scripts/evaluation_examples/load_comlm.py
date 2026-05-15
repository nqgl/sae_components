from comlm.exprank import XRNoisedBatch
from context import storage_name
from saeco_research.evaluation.evaluation import Evaluation

from saeco.mlog import mlog

mlog.init(project="markov-bio/evaluator")
root_eval = Evaluation[XRNoisedBatch].open_cache(storage_name)
print()
