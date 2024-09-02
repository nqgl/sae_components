# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
from saeco.trainer import Trainable

from saeco.architectures.anth_update import cfg, anth_update_model

from jaxtyping import Int, Float
from torch import Tensor
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
from functools import wraps
from saeco.evaluation.evaluation_context import Evaluation
import nnsight


ec = Evaluation.from_model_name(
    "L0Targeting/(lars)anth_update_model0.001[30.0]-101/50000"
)
# ec.store_acts(
#     CachingConfig(
#         store_dense=False,
#         dirname="ec_test2",
#         num_chunks=10,
#         docs_per_chunk=1000,
#         documents_per_micro_batch=16,
#     ),
#     displace_existing=True,
# )


# %%
# c = Chunk.load_chunks_from_dir(path, load_sparse_only=True)
# print(len(c))
from saeco.evaluation.saved_acts import SavedActs


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

from rich.highlighter import Highlighter

color = color_vecs[0]


def print_activity(
    tokens, feature_activity, document_id, feature_ids, colors=color_vecs
):
    # tokstrs = llm.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
    tokstrs = ec.llm.tokenizer._tokenizer.decode_batch(
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
                [f"{i:.02}" for i in fa.coalesce().values()],
                style=f"rgb({color[0]},{color[1]},{color[2]}) bold italic",
            )
    for i, t in enumerate(tokstrs):
        active: bool = False
        for j, fa in enumerate(feature_activity):
            if fa[i]:
                active = True
                color = colors[j]
                console.print(
                    t,
                    style=f"rgb({color[0]},{color[1]},{color[2]}) underline bold italic",
                    end="",
                    highlight=False,
                )
        if not active:
            console.print(t, style="rgb(255,255,255)", end="", highlight=False)
        # console.print(" ", end="")
    console.print("\n" * 5, highlight=False)


# def explain_colors():


ec = Evaluation.from_cache_name("ec_test2")


# %%


import tqdm

# for i in tqdm.trange(6144):
#     all_cosims.append(ec.cosims(ec.get_feature(i)))
# print()


import einops


# s = einops.einsum(feats, feats, "f d s, f2 d s -> f f2")


def select_active():

    feature_ids = [14, 15, 16, 17, 18, 19]
    feats = ec.get_features(feature_ids)
    top_feats = [ec.get_top_activating(f, percentile=10) for f in feats]
    fdocs = ec.select_active_documents(ec.features_union(top_feats))

    for i in fdocs.indices()[0]:
        print_activity(
            fdocs[i],
            [f[i] for f in top_feats],
            feature_ids=feature_ids,
            document_id=i,
        )


select_active()


def blocked(n_feats=6144, block_size=64):
    with torch.no_grad():
        memfeats = [ec.get_feature(i).cuda() for i in tqdm.trange(n_feats)]
        normed_feats = [m / m.values().pow(2).sum().sqrt() for m in memfeats]
        mat = torch.zeros(n_feats, n_feats).cuda()
        for i in tqdm.trange(0, n_feats, block_size):
            l_block = einops.rearrange(
                torch.stack(
                    [f.to_dense() for f in normed_feats[i : i + block_size]], dim=0
                ),
                "f d s -> f (d s)",
            )
            for j in range(0, n_feats, block_size):
                r_block = einops.rearrange(
                    torch.stack(
                        [f.to_dense() for f in normed_feats[j : j + block_size]], dim=0
                    ),
                    "f d s -> f (d s)",
                )
                mat[i : i + block_size, j : j + block_size] = (
                    l_block @ r_block.transpose(-2, -1)
                )
    return mat


def sequential(n_feats=6144):
    mat = torch.zeros(n_feats, n_feats).cuda()
    f2sum = torch.zeros(n_feats).cuda()
    for chunk in tqdm.tqdm(ec.saved_acts.chunks, total=len(ec.saved_acts.chunks)):
        acts = chunk.acts.cuda().to_dense()
        fds = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
        f2s = fds.pow(2).sum(-1)
        assert f2s.shape == (n_feats,)
        f2sum += f2s
        mat += fds @ fds.transpose(-2, -1)
    norms = f2sum.sqrt()
    mat /= norms.unsqueeze(0)
    mat /= norms.unsqueeze(1)
    return mat


# co1 = blocked()
# co2 = sequential()x

mc = ec.masked_activation_cosims()


def test_masked(masking_idx, masked, mc):
    f1 = ec.get_feature(masking_idx).cuda().to_dense().flatten()
    f2 = ec.get_feature(masked).cuda().to_dense().flatten()
    mask = f1 > 0

    f2[~mask] = 0
    f2 /= f2.pow(2).sum().sqrt()
    f1 /= f1.pow(2).sum().sqrt()
    mcs = (f1 * f2).sum()
    assert torch.allclose(mcs, mc[masking_idx, masked])


def logit_effect_of_feature(feature_id, documents: Int[Tensor, "doc seq"]):
    nnsight
