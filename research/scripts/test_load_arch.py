from saeco_research.evaluation import Evaluation

from saeco.architecture.sae_architecture import Architecture

path = "/home/g/workspace/saved_models/rand29/DynamicThreshSAE/51.arch_ref"
arch = Architecture.load(path, load_weights=True)

print()
e = Evaluation.from_model_path(path)
print("loaded_eval")
