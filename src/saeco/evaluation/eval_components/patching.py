from typing import TYPE_CHECKING

import einops
import nnsight
import torch
import tqdm

from saeco.evaluation.cached_artifacts import cache_version
from saeco.evaluation.fastapi_models.families_draft import (
    Family,
    FamilyLevel,
    FamilyRef,
    GetFamiliesResponse,
    ScoredFamilyRef,
    ScoredFeature,
)
from saeco.evaluation.fastapi_models.Feature import Feature
from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.utils import fwad_safe_sdp
from saeco.misc.nnsite import getsite, setsite
from torch import Tensor

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
    ):
        """
        patch_fn maps from acts to patched acts and returns the patched acts
        """

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
            if return_sae_acts:
                assert len(acts_l) == 1
                return (out, acts_l[0])
            return out

        if not for_nnsight:
            return call_sae

        def apply_nnsight(x):
            return nnsight.apply(call_sae, x)

        return apply_nnsight

    def run_with_sae(self: "Evaluation", tokens, patch=lambda x: x):
        with self.nnsight_model.trace(
            tokens, **self.sae_cfg.train_cfg.data_cfg.model_cfg.model_kwargs
        ) as tracer:
            lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
            res = self.sae_with_patch(patch, return_sae_acts=True)(lm_acts)
            patch_in = self._skip_bos_if_appropriate(lm_acts, res[0])
            setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
            out = self.nnsight_model.output.save()
        return out

    def forward_ad_with_sae(
        self: "Evaluation",
        tokens,
        tangent=None,
        tangent_gen=None,
        patch_fn=lambda x: x,
        return_prob_grads=False,
    ):
        with fwad_safe_sdp():
            assert tokens.ndim == 2
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
                with self.nnsight_model.trace(
                    tokens, **self.sae_cfg.train_cfg.data_cfg.model_cfg.model_kwargs
                ) as tracer:
                    lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
                    orig_lm_acts = lm_acts.save()
                    acts_re = patched_sae(orig_lm_acts).save()
                    patch_in = self._skip_bos_if_appropriate(lm_acts, acts_re)
                    setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
                    out = self.nnsight_model.output.save()
                    ls_logits = out.logits.log_softmax(dim=-1)
                    tangent = nnsight.apply(fwAD.unpack_dual, ls_logits).tangent.save()
                if return_prob_grads:
                    soft = out.logits.softmax(dim=-1)
                    probspace = fwAD.unpack_dual(soft).tangent
            if return_prob_grads:
                return out, tangent, probspace
            return out, tangent

    def _skip_bos_if_appropriate(self: "Evaluation", lm_acts, reconstructed_acts):
        if self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first:
            return torch.cat([lm_acts[:, :1], reconstructed_acts[:, 1:]], dim=1)
        return reconstructed_acts

    def patchdiff(
        self: "Evaluation", tokens, patch_fn, return_prob_diffs=False, invert=False
    ):
        normal = self.run_with_sae(tokens)
        patched = self.run_with_sae(tokens, patch_fn)
        diff = patched.logits.log_softmax(-1) - normal.logits.log_softmax(-1)
        if invert:
            diff = -diff
        if return_prob_diffs:
            probdiff = patched.logits.softmax(-1) - normal.logits.softmax(-1)
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

        with torch.no_grad():
            for chunk in tqdm.tqdm(self.saved_acts.chunks):
                tokens = chunk.tokens.to(self.cuda)
                docs, mask = tokens.index_where_valid(feature_active[0:1])
                seq_pos = feature_active[1, mask]
                assert docs.shape[0] == seq_pos.shape[0]
                for i in range(0, docs.shape[0], batch_size):
                    batch_docs = docs[i : i + batch_size]
                    batch_seq_pos = seq_pos[i : i + batch_size]

                    def patch_fn(acts):
                        acts = acts.clone()
                        acts[
                            torch.arange(batch_seq_pos.shape[0]),
                            batch_seq_pos,
                            feature_id,
                        ] *= scale
                        return acts

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
                            batch_docs,
                            tangent_gen=tangent_gen,
                            patch_fn=patch_fn,
                        )

                    else:
                        ldiff = self.patchdiff(batch_docs, patch_fn, invert=True)
                    yield ldiff, batch_seq_pos
