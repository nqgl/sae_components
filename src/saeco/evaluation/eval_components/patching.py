from typing import TYPE_CHECKING

import einops
import nnsight
import torch
import tqdm

from saeco.data.dict_batch import DictBatch
from saeco.evaluation.utils import fwad_safe_sdp
from saeco.misc.nnsite import getsite, setsite

if TYPE_CHECKING:
    from ..evaluation import Evaluation
import torch.autograd.forward_ad as fwAD


class Patching:
    def sae_with_patch(
        self: "Evaluation",
        patch_fn,
        for_nnsight=True,
        cache_template=None,
        call_patch_with_cache=False,
        act_name=None,
        return_sae_acts=False,
        return_sae_preacts=False,
    ):
        """
        patch_fn maps from acts to patched acts and returns the patched acts
        """
        assert not (return_sae_preacts and not return_sae_acts)

        def shaped_hook(shape, return_acts_l=None):
            def acts_hook(cache, acts):
                """
                Act_name=None corresponds to the main activations, usually correct
                """
                if cache._parent is None:
                    return acts
                if cache.act_metrics_name is not act_name:
                    return acts
                acts = einops.rearrange(
                    acts, "(doc seq) dict -> doc seq dict", doc=shape[0], seq=shape[1]
                )
                out = (
                    patch_fn(acts, cache=cache)
                    if call_patch_with_cache
                    else patch_fn(acts)
                )
                if return_sae_acts:
                    return_acts_l.append(out)
                return einops.rearrange(out, "doc seq dict -> (doc seq) dict")

            return acts_hook

        def call_sae(x):
            acts_l = []
            assert len(acts_l) == 0
            cache = self.sae.make_cache()
            cache.acts = ...
            cache.act_metrics_name = ...
            if return_sae_preacts:
                cache.pre_acts = ...
            shape = x.shape
            x = einops.rearrange(x, "doc seq data -> (doc seq) data")
            if cache_template is not None:
                cache += cache_template
            if return_sae_acts:
                hook = shaped_hook(shape, acts_l)
            else:
                hook = shaped_hook(shape)
            cache.register_write_callback("acts", hook)
            out = self.sae(x, cache=cache)
            out = einops.rearrange(
                out, "(doc seq) data -> doc seq data", doc=shape[0], seq=shape[1]
            )
            if return_sae_preacts:
                pre_acts = einops.rearrange(
                    cache.pre_acts,
                    "(doc seq) data -> doc seq data",
                    doc=shape[0],
                    seq=shape[1],
                )
            cache.destruct()
            if return_sae_acts:
                assert len(acts_l) == 1
                if return_sae_preacts:
                    return (out, acts_l[0], pre_acts)
                return (out, acts_l[0])
            return out

        if not for_nnsight:
            return call_sae

        def apply_nnsight(x):
            return nnsight.apply(call_sae, x)

        return apply_nnsight

    def run_with_sae(
        self: "Evaluation",
        tokens_or_batch,
        patch=lambda x: x,
        doc_indices=None,
        metadata=None,
        batch=None,
        return_batch=False,
    ):
        batch = batch or self._build_model_batch(
            tokens_or_batch, doc_indices=doc_indices, metadata=metadata
        )
        with self.model_adapter.trace(self.nnsight_model, batch) as tracer:
            lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
            res = self.sae_with_patch(patch, return_sae_acts=True)(lm_acts)
            patch_in = self._skip_bos_if_appropriate(lm_acts, res[0])
            setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
            out = self.model_adapter.unwrap_output(self.nnsight_model.output.save())
        if return_batch:
            return out, batch
        return out

    def forward_ad_with_sae(
        self: "Evaluation",
        tokens_or_batch,
        tangent=None,
        tangent_gen=None,
        patch_fn=lambda x: x,
        return_prob_grads=False,
        doc_indices=None,
        metadata=None,
    ):
        with fwad_safe_sdp():
            if torch.is_tensor(tokens_or_batch):
                assert tokens_or_batch.ndim == 2
            assert tangent or tangent_gen
            if tangent:

                def fwad_hook(acts):
                    acts = patch_fn(acts)
                    return fwAD.make_dual(acts, tangent)

            else:

                def fwad_hook(acts):
                    acts = patch_fn(acts)
                    return fwAD.make_dual(acts, tangent_gen(acts))

            patched_sae = self.sae_with_patch(fwad_hook)
            with fwAD.dual_level():
                batch = self._build_model_batch(
                    tokens_or_batch, doc_indices=doc_indices, metadata=metadata
                )
                with self.model_adapter.trace(self.nnsight_model, batch) as tracer:
                    lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
                    orig_lm_acts = lm_acts.save()
                    acts_re = patched_sae(orig_lm_acts).save()
                    patch_in = self._skip_bos_if_appropriate(lm_acts, acts_re)
                    setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
                    out = self.model_adapter.unwrap_output(
                        self.nnsight_model.output.save()
                    )
                    logits = self.model_adapter.get_logits(out)
                    ls_logits = logits.log_softmax(dim=-1)
                    tangent = nnsight.apply(fwAD.unpack_dual, ls_logits).tangent.save()
                if return_prob_grads:
                    soft = logits.softmax(dim=-1)
                    probspace = fwAD.unpack_dual(soft).tangent
            if return_prob_grads:
                return out, tangent, probspace
            return out, tangent

    def _skip_bos_if_appropriate(self: "Evaluation", lm_acts, reconstructed_acts):
        if self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first:
            return torch.cat([lm_acts[:, :1], reconstructed_acts[:, 1:]], dim=1)
        return reconstructed_acts

    def patchdiff(
        self: "Evaluation",
        tokens,
        patch_fn,
        return_prob_diffs=False,
        invert=False,
        doc_indices=None,
        use_loss=False,
    ):
        normal, batch = self.run_with_sae(
            tokens, doc_indices=doc_indices, return_batch=True
        )
        patched, _ = self.run_with_sae(
            tokens, patch_fn, doc_indices=doc_indices, batch=batch, return_batch=True
        )

        if use_loss:
            normal_loss = self.model_adapter.compute_loss(
                self.subject_model, normal, batch
            )
            patched_loss = self.model_adapter.compute_loss(
                self.subject_model, patched, batch
            )
            diff = patched_loss - normal_loss
            if invert:
                diff = -diff
            return diff

        normal_logits = self.model_adapter.get_logits(normal)
        patched_logits = self.model_adapter.get_logits(patched)
        diff = patched_logits.log_softmax(-1) - normal_logits.log_softmax(-1)
        if invert:
            diff = -diff
        if return_prob_diffs:
            probdiff = patched_logits.softmax(-1) - normal_logits.softmax(-1)
            if invert:
                probdiff = -probdiff
            return diff, probdiff
        return diff

    def forward_token_attribution_to_features(self: "Evaluation", tokens, seq_range):
        assert tokens.ndim == 1 or tokens.ndim == 2 and tokens.shape[0] == 1
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.repeat(seq_range[1] - seq_range[0], 1, 1)

        def tangentize_embedding(embedding):
            assert embedding.ndim == 3
            return (
                fwAD.make_dual(
                    torch.ones_like(embedding[:, :, 0]).unsqueeze(-1),
                    torch.eye(embedding.shape[1]).unsqueeze(-1)[
                        seq_range[0] : seq_range[1]
                    ],
                )
                * embedding
            )

        with fwad_safe_sdp():
            with fwAD.dual_level():
                with self.nnsight_model.trace(
                    tokens, **self.sae_cfg.train_cfg.data_cfg.model_cfg.model_kwargs
                ) as tracer:
                    embed = self.nnsight_model.transformer.wte.output
                    self.nnsight_model.transformer.wte.output = nnsight.apply(
                        tangentize_embedding, embed
                    )
                    lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
                    res = self.sae_with_patch(lambda x: x, return_sae_acts=True)(
                        lm_acts
                    )
                    sae_acts = res[1]

                    acts_tangent = nnsight.apply(
                        fwAD.unpack_dual, sae_acts
                    ).tangent.save()
                    lm_acts.stop()
        return acts_tangent

    def average_patching_effect_on_dataset(
        self: "Evaluation",
        feature_id,
        batch_size=8,
        scale=None,
        by_fwad=False,
        random_subset_n=None,
    ):
        """
        this will be like,
            - on the filtered dataset,
            - where the feature occurs
                - (for each position),
        what happens when we ablate the feature?
        """
        results = torch.zeros(
            self.seq_len,
            self.d_vocab,
        ).to(self.cuda)
        num_batches = torch.zeros(self.seq_len).to(self.cuda) + 1e-6
        for ldiff, batch_seq_pos in self.patching_effect_on_dataset(
            feature_id=feature_id,
            batch_size=batch_size,
            scale=scale,
            by_fwad=by_fwad,
            random_subset_n=random_subset_n,
        ):
            for j in range(batch_seq_pos.shape[0]):
                results[: -batch_seq_pos[j]] += ldiff[j, batch_seq_pos[j] :]
                num_batches[: -batch_seq_pos[j]] += 1
        return results / num_batches.unsqueeze(-1)

    def average_aggregated_patching_effect_on_dataset(
        self: "Evaluation",
        feature_id,
        batch_size=8,
        scale=None,
        by_fwad=False,
        random_subset_n=None,
    ):
        """
        this will be like,
            - on the filtered dataset,
            - where the feature occurs
                - (for each position),
        what happens when we ablate the feature?
        """
        results = torch.zeros(
            self.d_vocab,
        ).to(self.cuda)
        num_batches = 0
        for ldiff, batch_seq_pos in self.patching_effect_on_dataset(
            feature_id=feature_id,
            batch_size=batch_size,
            scale=scale,
            by_fwad=by_fwad,
            random_subset_n=random_subset_n,
        ):
            results += ldiff.sum(dim=0).mean(dim=0)
            num_batches += ldiff.shape[0]
        return results / num_batches

    def custom_patching_effect_aggregation(
        self: "Evaluation",
        feature_id,
        logits_call,
        batch_size=8,
        scale=None,
        by_fwad=False,
        random_subset_n=None,
    ):
        num_batches = torch.zeros(self.seq_len) + 1e-6
        for ldiff, batch_seq_pos in self.patching_effect_on_dataset(
            feature_id=feature_id,
            batch_size=batch_size,
            scale=scale,
            by_fwad=by_fwad,
            random_subset_n=random_subset_n,
        ):
            if logits_call is not None:
                logits_call(ldiff, batch_seq_pos)
        return num_batches

    def patching_effect_on_dataset(
        self: "Evaluation",
        feature_id,
        batch_size=8,
        scale=None,
        by_fwad=False,
        random_subset_n=None,
    ):
        """
        this will be like,
            - on the filtered dataset,
            - where the feature occurs
                - (for each position),
            what happens when we ablate the feature? (or do fwad)

        random_subset_n selects documents, rather than positions
            this behavior may not be ideal if it overweighs certain documents
        """
        if scale is None:
            scale = 1 if by_fwad else 0
        feature0 = self.features[feature_id].to(self.cuda)
        feature = feature0.filter_inactive_docs()
        if random_subset_n:
            if (s := feature.filter.mask.sum()) > random_subset_n:
                new_mask = torch.zeros_like(feature.filter.mask, device=self.cuda)
                new_mask[feature.filter.mask] = (
                    torch.randperm(s, device=self.cuda) < random_subset_n
                )
                assert new_mask.sum() == random_subset_n
                new_feature = feature.mask_by_other(
                    new_mask, return_ft=True, presliced=True, value_like=False
                )
                # nfi = new_feature.indices()
                # fi = feature.indices().transpose(0, 1).tolist()
                # if nfi.numel() > 0:
                #     for i in nfi.transpose(0, 1).tolist():
                #         assert i in fi
                feature = new_feature
        feature_active = feature.indices()
        feature = feature.to_dense()

        def batch_iter(bbatch):
            for chunk in tqdm.tqdm(self.saved_acts.chunks):
                token_store = chunk.tokens_batch
                doc_tokens = None
                doc_ids = None
                seq_pos = None
                tokens = chunk.tokens.to(self.cuda)
                docs_i, mask_i = tokens.index_where_valid(feature_active[0:1])
                doc_ids_i = feature_active[0, mask_i]
                seq_pos_i = feature_active[1, mask_i]
                doc_tokens = (
                    docs_i if doc_tokens is None else torch.cat([doc_tokens, docs_i])
                )
                doc_ids = (
                    doc_ids_i if doc_ids is None else torch.cat([doc_ids, doc_ids_i])
                )
                seq_pos = (
                    seq_pos_i if seq_pos is None else torch.cat([seq_pos, seq_pos_i])
                )
                while doc_tokens is not None and doc_tokens.shape[0] >= bbatch:
                    yield (
                        token_store,
                        doc_tokens[:bbatch],
                        doc_ids[:bbatch],
                        seq_pos[:bbatch],
                    )
                    doc_tokens = (
                        doc_tokens[bbatch:] if doc_tokens.shape[0] > bbatch else None
                    )
                    doc_ids = (
                        doc_ids[bbatch:]
                        if doc_ids is not None and doc_ids.shape[0] > bbatch
                        else None
                    )
                    seq_pos = seq_pos[bbatch:] if seq_pos.shape[0] > bbatch else None
                if doc_tokens is not None and doc_tokens.shape[0] > 0:
                    yield token_store, doc_tokens, doc_ids, seq_pos

        with torch.no_grad():
            for token_store, docs, doc_ids, seq_pos in batch_iter(batch_size * 4):
                for i in range(0, docs.shape[0], batch_size):
                    batch_docs = docs[i : i + batch_size]
                    batch_doc_ids = (
                        doc_ids[i : i + batch_size] if doc_ids is not None else None
                    )
                    batch_seq_pos = seq_pos[i : i + batch_size]

                    def select_batch_tokens(ts, indices):
                        if isinstance(ts, DictBatch):
                            return ts.__class__.construct_with_other_data(
                                {
                                    k: v.index_select(indices, dim=0)
                                    for k, v in ts.items()
                                },
                                ts._get_other_dict(),
                            )
                        return ts.index_select(dim=0, index=indices)

                    def patch_fn(acts):
                        acts = acts.clone()
                        acts[
                            torch.arange(batch_seq_pos.shape[0]),
                            batch_seq_pos,
                            feature_id,
                        ] *= scale
                        return acts

                    if batch_doc_ids is None:
                        batch_doc_ids = torch.arange(
                            batch_docs.shape[0], device=batch_docs.device
                        )

                    batch_input = select_batch_tokens(
                        token_store.to(self.cuda), batch_doc_ids
                    )

                    if by_fwad:

                        def tangent_gen(acts):
                            tangent = torch.zeros_like(acts)
                            tangent[
                                torch.arange(batch_seq_pos.shape[0]),
                                batch_seq_pos,
                                feature_id,
                            ] = 1
                            assert tangent.sum() == batch_seq_pos.shape[0]
                            return tangent

                        def tangent_gen2(acts):
                            tangent = torch.zeros_like(acts)
                            tangent[
                                torch.arange(batch_seq_pos.shape[0]),
                                batch_seq_pos,
                                feature_id,
                            ] = 1
                            assert tangent.sum() == batch_seq_pos.shape[0]
                            tangent = tangent - tangent.mean(dim=-1, keepdim=True)
                            return tangent

                        def tangent_gen3(acts: Tensor):
                            tangent = torch.zeros_like(acts)
                            tangent[
                                torch.arange(batch_seq_pos.shape[0]),
                                batch_seq_pos,
                                feature_id,
                            ] = 1
                            assert tangent.sum() == batch_seq_pos.shape[0]
                            adj = tangent.sum(dim=-1, keepdim=True) / (
                                acts.count_nonzero(dim=-1)
                                .unsqueeze(-1)
                                .expand(-1, -1, acts.shape[-1])
                                + 1e-6
                            )
                            mask = acts > 0
                            tangent[mask] = tangent[mask] - adj[mask]
                            return tangent

                        if by_fwad == 2:
                            tangent_gen = tangent_gen2
                        if by_fwad == 3:
                            tangent_gen = tangent_gen3

                        out, ldiff = self.forward_ad_with_sae(
                            batch_input,
                            tangent_gen=tangent_gen,
                            patch_fn=patch_fn,
                            doc_indices=batch_doc_ids,
                        )

                    else:
                        ldiff = self.patchdiff(
                            batch_input,
                            patch_fn,
                            invert=True,
                            doc_indices=batch_doc_ids,
                        )
                    yield ldiff, batch_seq_pos

    def get_acts_with_intervention(
        self: "Evaluation",
        tokens,
        write_in_site,
        intervention_fn,
        doc_indices=None,
        metadata=None,
    ):
        batch = self._build_model_batch(
            tokens, doc_indices=doc_indices, metadata=metadata
        )
        with self.model_adapter.trace(self.nnsight_model, batch) as tracer:
            setsite(
                self.nnsight_model,
                write_in_site,
                nnsight.apply(
                    intervention_fn, getsite(self.nnsight_model, write_in_site)
                ),
            )

            lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
            actsx2 = self.sae_with_patch(
                lambda x: x, return_sae_acts=True, return_sae_preacts=True
            )(lm_acts)
            sae_acts = actsx2[1].save()
            sae_preacts = actsx2[2].save()
            lm_acts.stop()
        return sae_acts, sae_preacts

    @torch.no_grad()
    def ff_multi_feature(
        self: "Evaluation",
        tokens,
        feat_ids,
        offset=1,
        lerp=0.02,
        scale=1,
        set_or_add=0,
        write_in_site="transformer.h.0.input",
    ):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0).expand(feat_ids.shape[0], -1)
        else:
            assert tokens.ndim == 2 and tokens.shape[0] == feat_ids.shape[0]

        def patch_fn(acts):
            acts = acts.clone()
            if set_or_add == 0:
                if (
                    not (acts[torch.arange(len(feat_ids)), :, feat_ids] > 0)
                    .any(-1)
                    .all()
                ):
                    mismatch = (acts[torch.arange(len(feat_ids)), :, feat_ids] > 0).any(
                        -1
                    )
                    print(
                        f"mismatch: {mismatch.sum()} total empty, {feat_ids[mismatch]}"
                    )
            assert acts.ndim == 3
            if set_or_add > 0:
                acts[torch.arange(len(feat_ids)), :, feat_ids] += set_or_add
            else:
                acts[torch.arange(len(feat_ids)), :, feat_ids] = set_or_add
            return acts

        normal_out1 = self.run_with_sae(tokens)
        patched_out1 = self.run_with_sae(tokens, patch_fn)

        input_embedding = self.nnsight_model._model.get_input_embeddings()

        normal_logits = self.model_adapter.get_logits(normal_out1)
        patched_logits = self.model_adapter.get_logits(patched_out1)
        normal_emb = (normal_logits.softmax(-1)) @ input_embedding.weight
        patched_emb = (patched_logits.softmax(-1)) @ input_embedding.weight

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

        patched_acts, patched_preacts = self.get_acts_with_intervention(
            tokens, write_in_site, make_intervention(patched_emb)
        )
        normal_acts, normal_preacts = self.get_acts_with_intervention(
            tokens, write_in_site, make_intervention(normal_emb)
        )
        if set_or_add == 0:
            return (
                normal_acts - patched_acts,
                normal_preacts - patched_preacts,
            )
        else:
            return patched_acts - normal_acts, patched_preacts - normal_preacts
