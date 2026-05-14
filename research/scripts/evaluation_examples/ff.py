# %%
import nnsight
import torch
import tqdm
from load import root_eval
from torch import Tensor

from saeco.misc.nnsite import getsite, setsite

# b = root_eval.average_aggregated_patching_effect_on_dataset(22535, random_subset_n=200)
# print("did B")
# a = root_eval.average_aggregated_patching_effect_on_dataset(22535)


@torch.no_grad()
def procedure2_by_ablation_preacts(
    tokens, feat_id, offset=1, set_to=0, lerp=0.02, scale=1
):
    def patch_fn(acts):
        acts = acts.clone()
        print("shapE", acts.shape)
        assert (
            (
                acts[
                    :,
                    :,
                    feat_id,
                ]
                > 0
            )
            .any(-1)
            .all()
        )
        assert acts.ndim == 3
        acts[
            :,
            :,
            feat_id,
        ] = set_to

        return acts

    normal_out1 = root_eval.run_with_sae(tokens)
    patched_out1 = root_eval.run_with_sae(tokens, patch_fn)

    input_embedding = root_eval.nnsight_model._model.get_input_embeddings()
    normal_emb = (normal_out1.logits.softmax(-1)) @ input_embedding.weight
    patched_emb = (patched_out1.logits.softmax(-1)) @ input_embedding.weight

    write_in_site = "transformer.h.0.input"
    with root_eval.nnsight_model.trace(tokens) as tracer:
        write_current = getsite(root_eval.nnsight_model, write_in_site)
        replace = write_current.clone()
        if offset:
            replace[:, offset:].lerp_(patched_emb[:, :-offset], lerp)
        else:
            replace.lerp_(patched_emb, lerp)
        replace *= scale
        setsite(root_eval.nnsight_model, write_in_site, replace)

        lm_acts = getsite(root_eval.nnsight_model, root_eval.nnsight_site_name)
        actsx2 = nnsight.apply(
            lambda a: root_eval.sae.get_acts(a, pre_acts=True), lm_acts
        )
        sae_acts = actsx2[1].save()

    patched = sae_acts.value
    with root_eval.nnsight_model.trace(tokens) as tracer:
        write_current = getsite(root_eval.nnsight_model, write_in_site)
        replace = write_current.clone()
        if offset:
            replace[:, offset:].lerp_(normal_emb[:, :-offset], lerp)
        else:
            replace.lerp_(normal_emb, lerp)
        replace *= scale
        setsite(root_eval.nnsight_model, write_in_site, replace)

        lm_acts = getsite(root_eval.nnsight_model, root_eval.nnsight_site_name)
        actsx2 = nnsight.apply(
            lambda a: root_eval.sae.get_acts(a, pre_acts=True), lm_acts
        )
        sae_acts = actsx2[1].save()

    normal = sae_acts.value
    fd = normal - patched
    fd = fd.unsqueeze(0)

    return fd


@torch.no_grad()
def ff_multi_feature(tokens, feat_ids, offset=1, lerp=0.02, scale=1, set_to=0):
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0).expand(feat_ids.shape[0], -1)
    else:
        assert tokens.ndim == 2 and tokens.shape[0] == feat_ids.shape[0]

    def patch_fn(acts):
        acts = acts.clone()
        if set_to == 0:
            assert (acts[torch.arange(len(feat_ids)), :, feat_ids] > 0).any(-1).all()
        assert acts.ndim == 3
        acts[torch.arange(len(feat_ids)), :, feat_ids] = set_to
        return acts

    normal_out1 = root_eval.run_with_sae(tokens)
    patched_out1 = root_eval.run_with_sae(tokens, patch_fn)

    input_embedding = root_eval.nnsight_model._model.get_input_embeddings()

    normal_emb = (normal_out1.logits.softmax(-1)) @ input_embedding.weight
    patched_emb = (patched_out1.logits.softmax(-1)) @ input_embedding.weight

    write_in_site = "transformer.h.0.input"

    def make_intervention(patch_in_value):
        def intervention_fn(acts: Tensor):
            replace = acts.clone()
            if offset:
                replace[:, offset:].lerp_(patch_in_value[:, :-offset], lerp)
            else:
                replace.lerp_(patch_in_value, lerp)
            replace *= scale
            return replace

        return intervention_fn

    patched_acts, patched_preacts = get_acts_with_intervention(
        tokens, write_in_site, make_intervention(patched_emb)
    )
    normal_acts, normal_preacts = get_acts_with_intervention(
        tokens, write_in_site, make_intervention(normal_emb)
    )
    if set_to == 0:
        fd = normal_preacts - patched_preacts
    else:
        fd = normal_acts - patched_acts
    return fd


# %%
def ftk2(feat, set_to=0, doc_index=1, ndocs=20, **kwargs):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    print(feat_active[doc_index : doc_index + ndocs])
    print(feat_active[doc_index : doc_index + ndocs].shape)

    doc = root_eval.docs[feat_active[doc_index : doc_index + ndocs].squeeze(-1)]
    print("".join(root_eval.detokenize(doc)[0]))
    fab = procedure2_by_ablation_preacts(doc, feat, set_to=set_to, **kwargs)

    tk = fab.mean(0).mean(0).topk(10)
    return set([i.item() for i in tk.indices]), tk, topk(-fab)


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


m = get_ff_matrix_on_document()
m2 = get_ff_matrix_on_document(2)
p, a = m
m.cpu().abs().mean()
v = m.abs() > 2e-3
(m.sign() == m2.sign())[v].sum() / v.sum()
v.sum()
v.nonzero()
m[-1]

p.sum()
p.abs()
print()
m.sum()
f = p - p.transpose(0, 1)
f.abs().sum()
(p.sign() == p.transpose(0, 1).sign()).sum() / p.numel()
