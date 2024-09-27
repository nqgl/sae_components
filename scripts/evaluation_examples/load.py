import nnsight
from context import model_name, storage_name
from saeco.evaluation.evaluation import Evaluation


root_eval = Evaluation.from_cache_name(storage_name)
# ec.sae_cfg.train_cfg.data_cfg.model_cfg.model
# nnsight_model = nnsight.LanguageModel("google/gemma-2b", device_map="cuda")
# ec.nnsight_model = nnsight_model
print()
