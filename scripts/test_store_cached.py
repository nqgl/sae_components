# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
from saeco.trainer import Trainable
from saeco.architectures.anth_update import cfg, anth_update_model
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache

modelss = Path.home() / "workspace/saved_models/"


name = "sweep_None/(lars)anth_update_model0.001[30.3]-95_10000"


def load(cfg: BaseModel, model_fn, name):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    tr = TrainingRunner(cfg, model_fn, state_dict=torch.load(pt_path))
    # tr.trainable.load_state_dict()
    return tr


tr = load(cfg, anth_update_model, name)


# %%
from saeco.evaluation.cached_data import ActsCacher, Chunk

path = Path.home() / "workspace" / "cached_sae_acts" / "test"
# ccfg = CachingConfig(store_dense=True)
# acts_cacher = ActsCacher(ccfg, tr, None)
# acts_cacher.store_acts()


# %%
c = Chunk.load_chunks_from_dir(path, load_sparse_only=True)
print(len(c))
from saeco.evaluation.saved_acts import SavedActs


sa = SavedActs(path)
print([f for f in sa.where_feature_active([1])])
active_tensor = sa.where_feature_active_big_tensor([1])

# %%
# torch.ones().to_sparse(,)
import colorama
from rich.console import Console

console = Console()
console.print("Hello", style="rgb(175,0,255)")
values = [70, 10, 255]
color_vecs = [
    torch.tensor(
        [
            values[a],
            values[b],
            values[c],
        ]
    )
    for a in range(3)
    for b in range(3)
    for c in range(3)
    if (a != b) and (b != c) and (a != c)
]

# %%


active_idxs = active_tensor.coalesce().indices()
chunkids = active_idxs[0]
docids = active_idxs[1]

active_tensor_documents = torch.cat(sa.tokens[chunkids, docids], dim=0)


def index_sparse(tensor: torch.Tensor, index):
    tensor = tensor.coalesce()
    shape = tensor.shape
    ids = tensor.indices()
    values = tensor.values()
    mask = torch.ones_like(values, dtype=torch.bool)
    nonindexed_shapes = []
    nonindexed = []

    for i, idx in enumerate(index):
        if idx is ...:
            nonindexed_shapes.append(shape[i])
            nonindexed.append(i)
            continue
        elif isinstance(idx, slice):
            assert idx.step is None
            if idx.start is None and idx.stop is None:
                nonindexed_shapes.append(shape[i])
                nonindexed.append(i)
                continue
            else:
                assert False, "need to add sliced indices and shapes"
            # elif idx.start is None:
            #     mask &= ids[i] < idx.stop
            # elif idx.stop is None:
            #     mask &= ids[i] >= idx.start
            # else:
            #     mask &= (ids[i] >= idx.start) & (ids[i] < idx.stop)
        elif isinstance(idx, torch.Tensor) and idx.ndim >= 1:

            if tensor.dtype == torch.bool:
                ...
                assert False
            else:
                assert tensor.dtype == torch.int64 and tensor.ndim == 1
                assert (idx < shape[i]).all()
                match = (ids[i].unsqueeze(1) == idx.unsqueeze(0)).any(dim=1)
                mask &= match
                nonindexed_shapes.append(idx.shape[0])
                nonindexed.append(i)
            assert False
        else:
            mask &= ids[i] == idx
    return torch.sparse_coo_tensor(
        ids[:, mask][nonindexed],
        # torch.cat(
        #     [torch.arange(mask.count_nonzero()).unsqueeze(0), ],
        #     dim=0,
        # ),
        values[mask],
        [*nonindexed_shapes],
    )


llm = tr.cfg.train_cfg.data_cfg.model_cfg.model
from rich.highlighter import Highlighter

color = color_vecs[0]


class TokenHighlighter(Highlighter):
    def highlight(self, text):
        return text.stylize(f"rgb({color[0]},{color[1]},{color[2]})")


highlighter = TokenHighlighter()


def print_activity(tokens, feature_activity, color=color_vecs[5], features=[1]):
    # tokstrs = llm.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
    tokstrs = llm.tokenizer._tokenizer.decode_batch(
        [[t] for t in tokens],
        skip_special_tokens=True,
    )
    if feature_activity.coalesce().values().shape[0] > 1:
        print("multiple activations")

    for i, t in enumerate(tokstrs):
        if feature_activity[i]:
            console.print(
                t,
                style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                end="",
            )
        else:
            console.print(t, style="rgb(255,255,255)", end="")
        # console.print(" ", end="")
    console.print("\n")


feature = 1
for i in range(active_idxs.shape[1]):
    print_activity(
        active_tensor_documents[i],
        index_sparse(active_tensor, [chunkids[i], docids[i], ..., feature]),
    )


# %%
