# %%
import torch
import tqdm
from load import root_eval
from torch import Tensor


@torch.no_grad()
def get_ff_matrix_on_document(doc_index=1, batch_size=32):
    ffmat = torch.zeros(root_eval.d_dict, root_eval.d_dict, device="cuda")

    doc = root_eval.docs[torch.arange(doc_index, doc_index + 1)].squeeze(0)
    for i in tqdm.trange(0, root_eval.d_dict, batch_size):
        d = root_eval.ff_multi_feature(
            doc, torch.arange(i, min(i + batch_size, root_eval.d_dict)), set_or_add=1
        )
        # ffmat_acts[i : i + batch_size] = d[0].mean(1)
        ffmat[i : i + batch_size] = d[1].mean(1)
    return ffmat.cuda()


@torch.no_grad()
def get_active_ff_matrix_on_document(doc_index=1, batch_size=32, active_only=True):
    ffmat = torch.zeros(root_eval.d_dict, root_eval.d_dict, device="cuda")
    ffmat_preacts = torch.zeros(root_eval.d_dict, root_eval.d_dict, device="cuda")

    doc = root_eval.docs[torch.arange(doc_index, doc_index + 1)].squeeze(0)
    if active_only:
        acts = (
            root_eval.acts[torch.arange(doc_index, doc_index + 1)]
            .to(root_eval.cuda)
            .to_dense()
        )

        features = (acts.sum(0).sum(0) > 0).nonzero().squeeze()

        print("features", features, acts.shape)
    else:
        features = torch.arange(root_eval.d_dict)

    for i in tqdm.trange(0, len(features), batch_size):
        d = root_eval.ff_multi_feature(
            doc.clone(), features[i : i + batch_size], set_or_add=0
        )
        acts[:, :, features[i : i + batch_size]].sum(0).sum(0)
        o = root_eval.nnsight_model._model(doc.cuda())
        with root_eval.nnsight_model.trace(doc.unsqueeze(0).cuda().repeat(2, 1)):
            out = root_eval.nnsight_model.output.save()
        out.logits[0, 6].topk(5)
        o.logits[6].topk(5)
        o.shape
        (o.logits - out.logits[1]).abs().sum()
        # ffmat_acts[i : i + batch_size] = d[0].mean(1)
        ffmat[features[i : i + batch_size]] = d[0].mean(1)
        ffmat_preacts[features[i : i + batch_size]] = d[1].mean(1)
    return ffmat.cuda(), ffmat_preacts.cuda(), features


# %%
DOC = 11
root_eval.detokenize(root_eval.docs[torch.arange(DOC, DOC + 1)].squeeze())


# %%

i = DOC
print("i", i)
m, m_pre, active = get_active_ff_matrix_on_document(i)
del m, m_pre, active

# %%

# if a affects b then a gets importance from b's importance
# so if the mat is from,to, then
# we want mat @ nodes, eg "how much do you affect important others"


def pagerank(m: Tensor, n=1000):
    assert m.ndim == 2 and m.shape[0] == m.shape[1]
    if m.isnan().any() or m.isinf().any():
        m = m.clone()
        m[m.isnan() | m.isinf()] = 0
    nodes = torch.ones_like(m[0])
    for i in tqdm.trange(n):
        nodes = m @ nodes
        if i % 10 == 0:
            nodes /= nodes.sum()

    nodes /= nodes.sum()
    return nodes


absrank = pagerank(m.abs())
Trank = pagerank(m.abs().T)
# posrank = pagerank(m.relu())
# %%

for f in absrank.topk(10).indices.tolist()[1:]:
    print(
        f,
        root_eval.detokenize(
            root_eval.average_aggregated_patching_effect_on_dataset(
                f, random_subset_n=100
            )
            .topk(30)
            .indices
        ),
    )
# %%

for f in Trank.topk(3).indices.tolist():
    print(
        f,
        root_eval.detokenize(
            root_eval.average_aggregated_patching_effect_on_dataset(
                f, random_subset_n=100
            )
            .topk(10)
            .indices
        ),
    )

# %%
# %%

acts = root_eval.acts[torch.arange(DOC, DOC + 1)]
# %%
