# %%
from pathlib import Path
from saeco.trainer import Trainable
from saeco.architectures.threshgate.tg_model2 import cfg, multigate_sae
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache

modelss = Path.home() / "workspace/saved_models/"


name = "(lars)multigate_sae0.001[25.0]-8273_1001"


def load(cfg: BaseModel, model_fn, name):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    tr = TrainingRunner(cfg, model_fn)
    tr.trainable.load_state_dict(torch.load(pt_path))
    return tr


tr = load(cfg, multigate_sae, name)
model = tr.trainable
# %%
x = next(tr.data)
y = model(x)
# %%
(y - x).pow(2).mean()
cache = TrainCache()
cache.L2_loss = ...
model.loss(x, cache=cache)
cache.L2_loss

# %%
