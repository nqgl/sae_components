# %%
import torch
import tqdm
from load import root_eval
from torch import Tensor

from saeco.misc.nnsite import getsite, setsite

e = root_eval.average_aggregated_patching_effect_on_dataset(65)

# tokens = root_eval.docs[torch.arange(5, 7, 1)]
# feature = 42


# def patch_fn(acts):
#     acts = acts.clone()
#     acts[
#         :,
#         :,
#         feature,
#     ] += 0.1
#     return acts


# diff = root_eval.patchdiff(tokens, patch_fn)


# # def ff_aa(tokens):

# # %%
# ie = root_eval.nnsight_model._model.get_input_embeddings()
# # %%

# first_diff = (diff.softmax(-1) - 0.5 / diff.shape[-1]) @ ie.weight

# # %%
# torch.nn.Embedding
# # %%

# write_in = "transformer.h.0.input"
# with root_eval.nnsight_model.trace(tokens) as tracer:
#     write_current = getsite(root_eval.nnsight_model, write_in)
#     setsite(root_eval.nnsight_model, write_in, write_current + first_diff)
#     lm_acts = getsite(root_eval.nnsight_model, root_eval.nnsight_site_name)
#     res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(lm_acts)
#     sae_acts = res[1].save()
#     patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
#     setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
#     out = root_eval.nnsight_model.output.save()
# patched = sae_acts

# with root_eval.nnsight_model.trace(tokens) as tracer:
#     lm_acts = getsite(root_eval.nnsight_model, root_eval.nnsight_site_name)
#     res = root_eval.sae_with_patch(patch_fn=lambda x: x, return_sae_acts=True)(lm_acts)
#     sae_acts = res[1].save()
#     patch_in = root_eval._skip_bos_if_appropriate(lm_acts, res[0])
#     setsite(root_eval.nnsight_model, root_eval.nnsight_site_name, patch_in)
#     out = root_eval.nnsight_model.output.save()
# normal = sae_acts


# # %%

# fd = patched - normal
# # %%
# top = fd.mean(0).mean(0).topk(35)
# # %%

# top


def topk(fd, k=35):
    tk = fd.mean(0).mean(0).topk(k)
    return tk


# %%


@torch.no_grad()
def procedure(tokens, feat_id, offset=1):
    def patch_fn(acts):
        acts = acts.clone()
        acts[
            :,
            :,
            feat_id,
        ] += 0.5
        return acts

    diff = root_eval.patchdiff(tokens, patch_fn)
    input_embedding = root_eval.nnsight_model._model.get_input_embeddings()

    first_diff = (diff.softmax(-1)) @ input_embedding.weight

    write_in_site = "transformer.h.0.input"
    with root_eval.nnsight_model.trace(tokens) as tracer:
        write_current = getsite(root_eval.nnsight_model, write_in_site)
        replace = write_current.clone()
        if offset:
            replace[:, offset:] += first_diff[:, :-offset]
        else:
            replace += first_diff
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


def topk(fd, k=35):
    tk = fd.mean(0).mean(0).topk(k)
    return tk


# %%
if False:
    feat = 605
    fd1 = procedure(root_eval.docs[torch.arange(5, 6, 1)], feat)
    tk = topk(fd1)
    tk

    from dtest import logit_effects

    logit_effects(feat, k=20, random_subset_n=100)

    tk.indices

    # %%
    for i in range(10):
        logit_effects(tk.indices[i].item(), k=20, random_subset_n=100)


# fd1 = procedure(root_eval.docs[torch.arange(5, 7, 1)], 66)

# fd2 = procedure(root_eval.docs[torch.arange(7, 13, 1)], 66)

# topk(fd1)

# # %%

# topk(fd2)
# # %%
# fd1.shape
# # %%
# topk(procedure(torch.randint(5, 10000, (1, 128)), 66))
# # %%


# %%
feat = 605


# @torch.inference_mode()
@torch.no_grad()
def procedure_one_pos_per_batch(tokens: Tensor, feat_id, offset=1, batch_size=8):
    assert tokens.shape[0] == 1
    tokens = tokens.expand(batch_size, -1)
    l = []
    for i in tqdm.trange(0, tokens.shape[1], batch_size):

        def patch_fn(acts):
            assert acts.ndim == 3
            # acts = acts.clone()
            r = torch.arange(acts.shape[0])
            acts[
                r,
                r + i,
                feat_id,
            ] += 0.5
            return acts

        diff = root_eval.patchdiff(tokens, patch_fn)
        input_embedding = root_eval.nnsight_model._model.get_input_embeddings()

        first_diff = (diff.softmax(-1)) @ input_embedding.weight

        write_in_site = "transformer.h.0.input"
        with root_eval.nnsight_model.trace(tokens) as tracer:
            write_current = getsite(root_eval.nnsight_model, write_in_site)
            replace = write_current.clone()
            if offset:
                replace[:, offset:] += first_diff[:, :-offset]
            else:
                replace += first_diff
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
        # l.append(fd.mean(1))
        l.append(fd)
    return torch.cat(l)


fdp = procedure_one_pos_per_batch(root_eval.docs[torch.arange(5, 6, 1)], feat)
# %%

fd = procedure(root_eval.docs[torch.arange(5, 6, 1)], feat)

# %%
fdm = fd.mean(0).mean(0)
fdpm = fdp.mean(0).mean(0)
# %%

k1 = topk(fd)
# %%
k2 = topk(fdp)
# %%
m = k1.indices != k2.indices
# %%
k1.indices[m]
# %%
k2.indices[m]
# %%
k3 = k2 = topk(fdp[80:])

# %%
k2.indices == k3.indices
# %%
fdp2 = procedure_one_pos_per_batch(root_eval.docs[torch.arange(6, 7, 1)], feat)

# %%
k4 = topk(fdp2)
# %%
for i in k2.indices:
    print((k4.indices == i).any())

# %%


@torch.no_grad()
def procedure_by_ablation(tokens, feat_id, offset=1, set_to=0, ln=False):
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

    diff = root_eval.patchdiff(tokens, patch_fn, invert=True)
    print(diff)
    input_embedding = root_eval.nnsight_model._model.get_input_embeddings()

    if ln:
        first_diff = (
            torch.nn.functional.layer_norm(diff, diff.shape[-1:])
        ) @ input_embedding.weight
    else:
        first_diff = (diff.softmax(-1)) @ input_embedding.weight

    write_in_site = "transformer.h.0.input"
    with root_eval.nnsight_model.trace(tokens) as tracer:
        write_current = getsite(root_eval.nnsight_model, write_in_site)
        replace = write_current.clone()
        if offset:
            replace[:, offset:] = first_diff[:, :-offset]
        else:
            replace = first_diff
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


# %%

feature = root_eval.features[feat]
feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
# %%
doc = root_eval.docs[torch.arange(feat_active[1].item(), feat_active[1].item() + 1, 1)]

# %%
fab = procedure_by_ablation(
    doc,
    feat,
)
tk = topk(fab)
fab1 = procedure_by_ablation(doc, feat, set_to=1)
tk1 = topk(fab1)
fab_10 = procedure_by_ablation(doc, feat, set_to=-10)
tk_10 = topk(fab_10)
fab100 = procedure_by_ablation(doc, feat, set_to=100)
tk100 = topk(fab100)


# %%
def shared_top(tk1, tk2):
    l = []
    for i in tk1.indices:
        v = (tk2.indices == i).any()
        print(v)
        if v:
            l.append(i.item())
    return set(l)


shared_top(tk, tk1)

# %%
shared_top(tk, tk_10)
# %%
shared_top(tk, tk100)

# %%
feature.to_dense().value[1341]
# %%
feature.indices()


# %%
def feature_tk(feat, set_to=0, doc_index=1):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    doc = root_eval.docs[
        torch.arange(
            feat_active[doc_index].item(), feat_active[doc_index].item() + 1, 1
        )
    ]
    fab = procedure_by_ablation(
        doc,
        feat,
        set_to=set_to,
    )
    tk = topk(fab)
    return tk


def feature_tk_norm(feat, doc_index=1):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    doc = root_eval.docs[
        torch.arange(
            feat_active[doc_index].item(), feat_active[doc_index].item() + 1, 1
        )
    ]
    fab = procedure(
        doc,
        feat,
    )
    return topk(fab)


# %%
shared_top(feature_tk(605, set_to=0), feature_tk_norm(605))


# %%
s1 = shared_top(feature_tk(605, set_to=10), feature_tk(605, set_to=-10))

s2 = shared_top(feature_tk(606, set_to=10), feature_tk(606, set_to=-10))


# %%
s1 & s2


# %%
def getmagsets(feats):
    return [shared_top(feature_tk(f, set_to=0), feature_tk_norm(f)) for f in feats]


m = getmagsets(range(710, 720))

# %%
s = m[0]


for ms in m[1:]:
    s &= ms

# %%

top_features = root_eval.seq_activation_probs.cuda().topk(100)

# %%
(
    shared_top(feature_tk(605), top_features)
    & shared_top(feature_tk(606), top_features)
    & shared_top(feature_tk(607), top_features)
)


# %%


s & s2
# %%
freqset = {i.item() for i in tkfreq.indices}
# %%

s & freqset


# %%
def ftk(feat, set_to=0, doc_index=1, **kwargs):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    doc = root_eval.docs[
        torch.arange(
            feat_active[doc_index].item(), feat_active[doc_index].item() + 1, 1
        )
    ]
    fab = procedure_by_ablation(doc, feat, set_to=set_to, **kwargs)
    tk = topk(fab)
    return set([i.item() for i in topk(fab).indices])


def ftkn(feat, doc_index=1, **kwargs):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    doc = root_eval.docs[
        torch.arange(
            feat_active[doc_index].item(), feat_active[doc_index].item() + 1, 1
        )
    ]
    fab = procedure(doc, feat, **kwargs)
    return set([i.item() for i in topk(fab).indices])


# %%
ftk(605)
# %%


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
    fd = normal - patched
    return fd


# %%
def ftk2(feat, set_to=0, doc_index=1, ndocs=20, **kwargs):
    feature = root_eval.features[feat]
    feat_active = feature.filter_inactive_docs().filter.mask.nonzero()
    print(feat_active[doc_index : doc_index + ndocs])
    print(feat_active[doc_index : doc_index + ndocs].shape)

    doc = root_eval.docs[feat_active[doc_index : doc_index + ndocs].squeeze(-1)]
    print("".join(root_eval.detokenize(doc)[0]))
    fab = procedure2_by_ablation(doc, feat, set_to=set_to, **kwargs)
    tk = topk(fab)
    return set([i.item() for i in topk(fab).indices]), tk


s22, tk22 = ftk2(605)
ftk2(605)


# %%
e = root_eval.average_aggregated_patching_effect_on_dataset(65)

# %%
root_eval.detokenize(e.topk(50).indices)
# %%
for i in [j.item() for j in tk22.indices]:
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
