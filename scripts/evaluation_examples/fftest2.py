# %%
import nnsight
import torch
import tqdm
from load import root_eval
from saeco.evaluation.fastapi_models.feature_effects import (
    FeatureLogitEffectsRequest,
    TopKFeatureEffects,
)
from saeco.misc.nnsite import getsite, setsite
from torch import Tensor

b = root_eval.average_aggregated_patching_effect_on_dataset(22535, random_subset_n=200)
print("did B")
a = root_eval.average_aggregated_patching_effect_on_dataset(22535)


def topk(fd, k=35):
    tk = fd.mean(0).mean(0).topk(k)
    return tk


@torch.no_grad()
def procedure2_by_ablation(
    tokens, feat_id, offset=1, set_to=0, ln=False, lerp=0.5, scale=1.5
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
        res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(
            lm_acts
        )
        sae_acts = res[1].save()
        patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
        setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
        out = root_eval.nnsight_model.output.save()
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
        res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(
            lm_acts
        )
        sae_acts = res[1].save()
        patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
        setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
        out = root_eval.nnsight_model.output.save()

    normal = sae_acts.value
    fd = patched - normal
    return fd


@torch.no_grad()
def procedure2_by_ablation_preacts(
    tokens, feat_id, offset=1, set_to=0, ln=False, lerp=0.03, scale=1
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
    fd = patched - normal
    fd = fd.unsqueeze(0)

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
    print(fab)
    print(fab.shape)
    tk = topk(fab)
    return set([i.item() for i in tk.indices]), tk, topk(-fab)


s22, tk22, ntk = ftk2(605, doc_index=1, ndocs=10, set_to=0)


# %%
e = root_eval.average_aggregated_patching_effect_on_dataset(605)
tk22
# %%
root_eval.detokenize(e.topk(50).indices)
# %%

for i in [j.item() for j in tk22.indices[:3]]:
    print(i)
    print(
        ",".join(
            root_eval.detokenize(
                root_eval.average_aggregated_patching_effect_on_dataset(
                    i, random_subset_n=200
                )
                .topk(30)
                .indices
            )
        )
    )
print("INVERSES")
for i in [j.item() for j in ntk.indices[:3]]:
    print(i)
    print(
        ",".join(
            root_eval.detokenize(
                root_eval.average_aggregated_patching_effect_on_dataset(
                    i, random_subset_n=200
                )
                .topk(30)
                .indices
            )
        )
    )

# %%


@torch.no_grad()
def procedure2(tokens, feat_id, offset=1, set_to=0, ln=False, lerp=0.5, scale=1.5):
    def patch_fn(acts):
        acts = acts.clone()
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
        res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(
            lm_acts
        )
        sae_acts = res[1].save()
        patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
        setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
        out = root_eval.nnsight_model.output.save()
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
        res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(
            lm_acts
        )
        sae_acts = res[1].save()
        patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
        setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
        out = root_eval.nnsight_model.output.save()

    normal = sae_acts.value
    fd = normal - patched
    return fd


# %%


query = FeatureLogitEffectsRequest(
    feature=605,
)

ev = query.filter(root_eval)
effects = ev.average_aggregated_patching_effect_on_dataset(
    feature_id=query.feature,
    by_fwad=query.by_fwad,
    random_subset_n=query.random_subset_n,
)
topk = effects.topk(query.k)
# %%
TopKFeatureEffects(
    tokens=ev.detokenize(topk.indices),
    values=topk.values,
)

# %%
root_eval.doc_activation_probs.topk(3)

# %%
