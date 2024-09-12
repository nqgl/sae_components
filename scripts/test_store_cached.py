# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.storage.chunk import Chunk
from saeco.trainer import Trainable

# from saeco.architectures.anth_update import cfg, anth_update_model
from saeco.architectures.threshgate_gradjust.tg_grad_deep_model import (
    cfg,
    deep_tg_grad_sae,
)
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
from functools import wraps


def timed(func, name=""):
    import time

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{name}{func.__name__} took {end - start} seconds")
        return result

    return wrapped


modelss = Path.home() / "workspace/saved_models/"


name = "sweep_None/(lars)anth_update_model0.001[30.3]-95_10000"
name = "sweep_None/(lars)deep_tg_grad_sae0.003[25.0]-31_10000"


def load(cfg: BaseModel, model_fn, name):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    tr = TrainingRunner(cfg, model_fn)
    tr.trainable.load_state_dict(torch.load(pt_path))
    return tr


tr = load(cfg, deep_tg_grad_sae, name)
tr.trainable.eval()

# %%
from saeco.evaluation.acts_cacher import ActsCacher, Chunk


# path = Path.home() / "worksp    ace" / "cached_sae_acts" / "test"
# ccfg = CachingConfig(store_dense=True)
# acts_cacher = ActsCacher(ccfg, tr, None)
# acts_cacher.store_acts()

import time


@timed
def re_store(cfg):
    ac = ActsCacher(
        cfg,
        tr,
        None,
    )
    # remove the old store
    if ac.path().exists():
        ac.path().rename(
            ac.path().parent / "old" / f"old_{time.time()}{ac.path().name}"
        )
    ac.store_acts()
    return ac


# re_store(
#     CachingConfig(
#         store_dense=False, dirname="many_chunk", num_chunks=10, docs_per_chunk=100
#     ),
# )
# re_store(
#     CachingConfig(
#         store_dense=False,
#         dirname="tiny_chunk",
#         num_chunks=1000,
#         docs_per_chunk=10,
#         documents_per_micro_batch=8,
#     ),
# )
# path_tiny = Path.home() / "workspace" / "cached_sae_acts" / "tiny_chunk"

# re_store(
#     CachingConfig(
#         store_dense=False,
#         dirname="small_chunk",
#         num_chunks=1000,
#         docs_per_chunk=100,
#         documents_per_micro_batch=8,
#     ),
# )
# path_small = Path.home() / "workspace" / "cached_sae_acts" / "small_chunk"


# re_store(
#     CachingConfig(
#         store_dense=False,
#         dirname="big_chunk",
#         num_chunks=100,
#         docs_per_chunk=1000,
#         documents_per_micro_batch=8,
#     ),
# )
# path_big = Path.home() / "workspace" / "cached_sae_acts" / "big_chunk"

# re_store(
#     CachingConfig(
#         store_dense=False,
#         dirname="feat_store",
#         num_chunks=10,
#         docs_per_chunk=100,
#         documents_per_micro_batch=8,
#     ),
# )
path_big = Path.home() / "workspace" / "cached_sae_acts" / "feat_store"
# %%
# c = Chunk.load_chunks_from_dir(path, load_sparse_only=True)
# print(len(c))
from saeco.evaluation.saved_acts import SavedActs

tr.trainable.model.model.module.freqs.freqs

sa = SavedActs(path_big)
feature_ids = [213]
# test_get_feature = timed(sa.active_feature_tensor)(213)


@timed
def load_feat_to_cuda():
    t = sa.active_feature_tensor(213).cuda()
    torch.cuda.synchronize()
    # t = t + 1
    return t


@timed
def find_overlap_cuda(features=[213, 214]):
    t = sa.active_feature_tensor(features[0]).cuda()
    for f in features[1:]:
        t = t * sa.active_feature_tensor(f).cuda()
    return t.coalesce()


@timed
def find_overlap(features=[213, 214]):
    t = sa.active_feature_tensor(features[0])
    for f in features[1:]:
        t = t * sa.active_feature_tensor(f)
    return t.coalesce()


load_feat_to_cuda()

active_tensor = timed(sa.where_feature_active_big_tensor, "big")(feature_ids)
active_idxs = active_tensor.coalesce().indices()
active_docs = active_idxs[0:2].unique(dim=1)
chunkids = active_docs[0]
docids = active_docs[1]

active_tensor_documents = torch.cat(
    timed(sa.ctokens.__getitem__, "big")((chunkids, docids)), dim=0
)

# sa = SavedActs(path_small)
# feature_ids = [213, 214, 215, 216, 217, 218]
# active_tensor = timed(sa.where_feature_active_big_tensor, "small")(feature_ids)
# active_idxs = active_tensor.coalesce().indices()
# active_docs = active_idxs[0:2].unique(dim=1)
# chunkids = active_docs[0]
# docids = active_docs[1]

# active_tensor_documents = torch.cat(
#     timed(sa.tokens.__getitem__, "small")((chunkids, docids)), dim=0
# )

# sa = SavedActs(path_tiny)
# feature_ids = [213, 214, 215, 216, 217, 218]
# active_tensor = timed(sa.where_feature_active_big_tensor, "tiny")(feature_ids)
# active_idxs = active_tensor.coalesce().indices()
# active_docs = active_idxs[0:2].unique(dim=1)
# chunkids = active_docs[0]
# docids = active_docs[1]

# active_tensor_documents = torch.cat(
#     timed(sa.tokens.__getitem__, "tiny")((chunkids, docids)), dim=0
# )


# (sa.chunks[0].acts.indices()[2] == 1)
# ids = sa.chunks[0].acts.indices()
# intersection = False
# feat_ids = ids[2]
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
        skip_special_tokens=False,
    )
    # if feature_activity.coalesce().values().shape[0] > 1:
    #     print("multiple activations", feature_activity.coalesce().values().shape[0])

    for i, t in enumerate(tokstrs):
        active: bool = False
        for j, fa in enumerate(feature_activity):
            if fa[i]:
                active = True
                color = color_vecs[j]
                console.print(
                    t,
                    style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                    end="",
                )
        if not active:
            console.print(t, style="rgb(255,255,255)", end="")
        # console.print(" ", end="")
    console.print("\n")


for i in range(active_idxs.shape[1]):
    print_activity(
        active_tensor_documents[i],
        [
            index_sparse(active_tensor, [chunkids[i], docids[i], ..., feature_id])
            for feature_id in feature_ids
        ],
    )
i = 2
index_sparse(
    active_tensor,
    [
        chunkids[i],
        docids[i],
        ...,
    ],
)
# %%
