import nnsight
from context import model_name, storage_name
from saeco.evaluation.evaluation import Evaluation


root_eval = Evaluation.from_cache_name(storage_name)
