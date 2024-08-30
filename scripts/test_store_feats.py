# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
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
# name = "sweep_None/(lars)deep_tg_grad_sae0.003[25.0]-31_10000"
name = "sweep_None/(lars)deep_tg_grad_sae0.002[25.0]-66_25000"
name = "sweep_f6h6fg5m/(lars)deep_tg_grad_sae0.001[512.0]-40_100001"
name = "binarize/(lars)deep_tg_grad_sae0.0003[100]-74_100001"


def load(cfg: BaseModel, model_fn, name):
    modelss: Path = Path.home() / "workspace/saved_models/"
    pt_path = modelss / (name + ".pt")
    cfg_path = modelss / (name + ".json")
    cfg = cfg.model_validate_json(cfg_path.read_text())
    from llm_finetune import tr

    state = torch.load(pt_path)
    # for k in list(state.keys()):
    #     if "proj_in.wrapped" in k:
    #         state[k.replace("proj_in.wrapped", "proj_in")] = state.pop(k)
    #     if "proj_out.wrapped" in k:
    #         state[k.replace("proj_out.wrapped", "proj_out")] = state.pop(k)
    tr.trainable.load_state_dict(state)
    return tr


tr = load(cfg, deep_tg_grad_sae, name)
# tr = TrainingRunner(cfg, model_fn=deep_tg_grad_sae)
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
        old = ac.path().parent / "old"
        old.mkdir(exist_ok=True, parents=True)
        ac.path().rename(old / f"old_{time.time()}{ac.path().name}")
    ac.store_acts()
    return ac


# re_store(
#     CachingConfig(
#         store_dense=False,
#         dirname="feat_store2",
#         num_chunks=2,
#         docs_per_chunk=50,
#         documents_per_micro_batch=16,
#     ),
# )

path = Path.home() / "workspace" / "cached_sae_acts" / "feat_store2"
# %%
# c = Chunk.load_chunks_from_dir(path, load_sparse_only=True)
# print(len(c))
from saeco.evaluation.saved_acts import SavedActs

# tr.trainable.model.model.module.freqs.freqs

sa = SavedActs(path)


@timed
def get_features_and_active_docs(feature_ids, intersection=False):
    feature_tensors = [sa.active_feature_tensor(fid) for fid in feature_ids]
    # active_documents_l = [
    #     f.indices()[0, f.values() != 0].unique() for f in feature_tensors
    # ]
    f = feature_tensors[0].clone()
    for ft in feature_tensors[1:]:
        if intersection:
            f = f * ft
        else:
            f += ft
    assert f.is_sparse
    f = f.coalesce()
    active_documents_idxs = f.indices()[0][f.values() != 0].unique()
    active_documents = sa.tokens[active_documents_idxs.unsqueeze(0)]
    if intersection:
        feature_tensors = [f]
    return feature_tensors, active_documents, active_documents_idxs


1, 2, 4
feature_ids = [14, 15, 16, 17, 18, 19]

feats, docs, doc_ids = get_features_and_active_docs(feature_ids, intersection=False)


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
# torch.ones().to_sparse(,)ts[import colorama
from rich.console import Console

console = Console()
console.print("Hello", style="rgb(175,0,255)")
values = [110, 15, 240]
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

llm = tr.cfg.train_cfg.data_cfg.model_cfg.model
from rich.highlighter import Highlighter

color = color_vecs[0]


def print_activity(tokens, feature_activity, document_id, colors=color_vecs):
    # tokstrs = llm.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
    tokstrs = llm.tokenizer._tokenizer.decode_batch(
        [[t] for t in tokens],
        skip_special_tokens=False,
    )

    # if feature_activity.coalesce().values().shape[0] > 1:
    #     print("multiple activations", feature_activity.coalesce().values().shape[0])
    console.print(
        f"\n\n\n\nDocument {document_id}", style="underline bold", highlight=False
    )
    console.print("\n" + "-" * 30 + "\n", highlight=False)
    for j, fa in enumerate(feature_activity):
        if fa.any():
            color = colors[j]
            console.print(
                f"Feature {feature_ids[j]} active",
                # [f"{i:.02}" for i in fa.coalesce().values()],
                style=f"rgb({color[0]},{color[1]},{color[2]}) bold italic",
            )
    for i, t in enumerate(tokstrs):
        active: bool = False
        for j, fa in enumerate(feature_activity):
            if fa[i]:
                color = colors[j]
                if not active:
                    console.print(
                        "[",
                        style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                        end="",
                        highlight=False,
                    )
                # else:
                #     console.print(
                #         f"[+{j}]",
                #         style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                #         end="",
                #         highlight=False,
                #     )

                # active = True
        if not active:
            console.print(t, style="rgb(255,255,255)", end="", highlight=False)
        for j, fa in enumerate(feature_activity):
            if fa[i]:
                color = colors[j]
                if not active:
                    console.print(
                        "]",
                        style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                        end="",
                        highlight=False,
                    )

        # console.print(" ", end="")
    console.print("\n" * 5, highlight=False)


# def explain_colors():


for i in range(docs.shape[0]):
    print_activity(
        docs[i],
        [f[doc_ids[i]] for f in feats],
        document_id=doc_ids[i],
    )

# %%
