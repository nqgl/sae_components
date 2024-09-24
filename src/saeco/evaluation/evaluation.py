from functools import cached_property
from pathlib import Path

import einops
import nnsight
import torch
import torch.autograd.forward_ad as fwAD
import tqdm
from attr import define, field
from torch import Tensor

from saeco.trainer import RunConfig, TrainingRunner
from .acts_cacher import ActsCacher, CachingConfig
from .filtered import FilteredTensor
from .metadata import MetaDatas
from .nnsite import getsite, setsite, tlsite_to_nnsite
from .saved_acts import SavedActs


@define
class Evaluation:
    model_name: str
    training_runner: TrainingRunner = field(repr=False)
    saved_acts: SavedActs | None = field(default=None, repr=False)
    # cache_name: str | None = None
    # cache_path: Path | None = None
    nnsight_model: nnsight.LanguageModel | None = field(default=None, repr=False)
    # metadatas: MetaDatas = field(init=False)

    @cached_property
    def metadatas(self):
        return MetaDatas(self.path, self.cache_cfg)

    @classmethod
    def from_cache_name(cls, name: Path | str):
        if isinstance(name, str):
            name = Path(name)
            if not name.exists():
                name = Path.home() / "workspace" / "cached_sae_acts" / name
        saved = SavedActs.from_path(name)
        inst = cls.from_model_name(saved.cfg.model_name)
        inst.saved_acts = saved
        return inst

    @classmethod
    def from_model_name(cls, name: str):
        tr = TrainingRunner.autoload(name)
        inst = cls(training_runner=tr, model_name=name)
        return inst

    def store_acts(self, caching_cfg: CachingConfig, displace_existing=False):
        if caching_cfg.model_name is None:
            caching_cfg.model_name = self.model_name
        assert caching_cfg.model_name == self.model_name
        acts_cacher = ActsCacher(
            caching_config=caching_cfg, model_context=self.training_runner
        )
        if acts_cacher.path().exists():
            if displace_existing:
                import time

                old = acts_cacher.path().parent / "old"
                old.mkdir(exist_ok=True, parents=True)
                acts_cacher.path().rename(
                    old / f"old_{time.time()}{acts_cacher.path().name}"
                )
            else:
                raise FileExistsError(
                    f"{acts_cacher.path()} already exists. Set displace_existing=True to move existing files."
                )
        acts_cacher.store_acts()
        self.saved_acts = SavedActs.from_path(acts_cacher.path())

    @property
    def d_dict(self):
        return self.training_runner.cfg.init_cfg.d_dict

    @property
    def sae_cfg(self) -> RunConfig:
        return self.training_runner.cfg

    @property
    def cache_cfg(self) -> CachingConfig:
        return self.saved_acts.cfg

    @property
    def path(self):
        if self.saved_acts is None:
            raise ValueError("cache_name must be set")
        return self.cache_cfg.path

    def get_active_documents(self, feature_ids):
        features = self.get_features(feature_ids)
        return self.select_active_documents(self.features_union(features))

    def get_features(self, feature_ids):
        return [self.features[fid] for fid in feature_ids]

    def get_feature(self, feature_id) -> FilteredTensor:
        return self.features[feature_id]

    @staticmethod
    def features_union(feature_tensors):
        f = feature_tensors[0].clone()
        for ft in feature_tensors[1:]:
            f += ft
        assert f.is_sparse
        f = f.coalesce()
        return f

    @staticmethod
    def active_document_indices(feature):
        return feature.indices()[0][feature.values() != 0].unique()

    def select_active_documents(self, feature):
        docs, doc_ids = self._active_docs_values_and_indices(feature)
        assert not docs.is_sparse
        return torch.sparse_coo_tensor(
            indices=doc_ids.unsqueeze(0),
            values=docs,
            size=(self.cache_cfg.num_docs, docs.shape[1]),
        ).coalesce()

    def filter_docs(self, docs_filter, only_return_selected=False, seq_level=False):
        ###
        if not only_return_selected:
            if seq_level:
                mask = torch.zeros(
                    self.cache_cfg.num_docs,
                    self.sae_cfg.train_cfg.data_cfg.seq_len,
                    dtype=torch.bool,
                )

            else:
                mask = torch.zeros(self.cache_cfg.num_docs, dtype=torch.bool)
        values = []
        for chunk in self.saved_acts.chunks:
            tokens = chunk.tokens.to(docs_filter.device)
            filt_docs = tokens.mask_by_other(
                docs_filter, return_ft=True, presliced=False
            )
            values.append(filt_docs.value)
            if not only_return_selected:
                filt_docs.filter.slice(mask)[:] = filt_docs.filter.mask
        values = torch.cat(values, dim=0)
        if only_return_selected:
            return values
        return FilteredTensor.from_value_and_mask(value=values, mask=mask)

    def filter_acts(self, docs_filter, only_return_selected=False):
        ###
        if not only_return_selected:
            mask = torch.zeros(self.cache_cfg.num_docs, dtype=torch.bool)
        values = []
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.to(docs_filter.device)
            filt = acts.filter.slice(docs_filter)
            if not filt.any():
                continue
            filt_docs = acts.mask_by_other(filt, return_ft=True, presliced=True)
            values.append(filt_docs.value)
            if not only_return_selected:
                filt_docs.filter.slice(mask)[:] = filt_docs.filter.mask
        values = torch.cat(values, dim=0)
        if only_return_selected:
            return values
        return FilteredTensor.from_value_and_mask(value=values, mask=mask)

    def _active_docs_values_and_indices(self, feature):
        active_documents_idxs = self.active_document_indices(feature)
        active_documents = self.saved_acts.tokens[active_documents_idxs]
        return active_documents, active_documents_idxs

    def top_activating_examples(self, feature_id: int, proportion=0.1):
        assert 0 < proportion and proportion < 1
        feature = self.features[feature_id]
        top = self._get_top_activating(feature.value, proportion=proportion)
        return feature.to_filtered_like_self(top)

    @property
    def features(self):
        return self.saved_acts.features

    @staticmethod
    def _get_top_activating(feature: Tensor, proportion=None, percentile=None):
        assert proportion is None or percentile is None
        proportion = proportion or (percentile / 100)
        topk = feature.values().topk(int(feature.values().shape[0] * proportion))
        return torch.sparse_coo_tensor(
            feature.indices()[:, topk.indices],
            topk.values,
            feature.shape,
        )

    @property
    def llm(self):
        return self.training_runner.cfg.train_cfg.data_cfg.model_cfg.model

    @property
    def sae(self):
        return self.training_runner.trainable

    def activation_cosims(self):
        mat = torch.zeros(self.d_dict, self.d_dict).cuda()
        f2sum = torch.zeros(self.d_dict).cuda()
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.cuda().to_dense()
            assert acts.ndim == 3
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum = f2s + f2sum
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def masked_activation_cosims(self):
        """
        Returns the masked cosine similarities matrix.
        Indexes are like: [masking feature, masked feature]
        """
        threshold = 0
        mat = torch.zeros(self.d_dict, self.d_dict).cuda()
        f2sum = torch.zeros(self.d_dict).cuda()
        maskedf2sum = torch.zeros(self.d_dict, self.d_dict).cuda()
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.cuda().to_dense()
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            feats_mask = feats_mat > threshold

            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum += f2s
            maskedf2sum += feats_mask.float() @ feats_mat.transpose(-2, -1).pow(2)
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= maskedf2sum.sqrt()
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def co_occurrence(self, pooling="mean"):
        """
        Pooling: "mean", "max" or "binary"
        this could be done at sequence level if we want
        """
        threshold = 0
        mat = torch.zeros(self.d_dict, self.d_dict).cuda()
        f2sum = torch.zeros(self.d_dict).cuda()
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.cuda().to_dense()
            assert acts.ndim == 3
            if pooling == "mean":
                acts_pooled = acts.mean(1)
            elif pooling == "max":
                acts_pooled = acts.max(1).values
            elif pooling == "binary":
                acts_pooled = (acts > threshold).sum(1).float()
            feats_mat = einops.rearrange(acts_pooled, "doc feat -> feat doc")
            f2s = feats_mat.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum = f2s + f2sum
            mat += feats_mat @ feats_mat.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def sae_with_patch(
        self,
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

    def run_with_sae(self, tokens, patch=lambda x: x):
        with self.nnsight_model.trace(tokens) as tracer:
            lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
            res = self.sae_with_patch(patch, return_sae_acts=True)(lm_acts)
            patch_in = self._skip_bos_if_appropriate(lm_acts, res[0])
            setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
            out = self.nnsight_model.output.save()
        return out

    @property
    def nnsight_site_name(self):
        return tlsite_to_nnsite(
            self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site
        )

    def forward_ad_with_sae(
        self,
        tokens,
        tangent=None,
        tangent_gen=None,
        patch_fn=lambda x: x,
        return_prob_grads=False,
    ):
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
            with self.nnsight_model.trace(tokens) as tracer:
                lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
                orig_lm_acts = lm_acts.save()
                acts_re = patched_sae(orig_lm_acts).save()
                patch_in = self._skip_bos_if_appropriate(lm_acts, acts_re)
                setsite(self.nnsight_model, self.nnsight_site_name, patch_in)
                out = self.nnsight_model.output.save()
                tangent = nnsight.apply(fwAD.unpack_dual, out.logits).tangent.save()
            if return_prob_grads:
                soft = out.logits.softmax(-1)
                probspace = fwAD.unpack_dual(soft).tangent
        if return_prob_grads:
            return out, tangent, probspace
        return out, tangent

    def _skip_bos_if_appropriate(self, lm_acts, reconstructed_acts):
        if self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first:
            return torch.cat([lm_acts[:, :1], reconstructed_acts[:, 1:]], dim=1)
        return reconstructed_acts

    def patchdiff(self, tokens, patch_fn, return_prob_diffs=False):
        normal = self.run_with_sae(tokens)
        patched = self.run_with_sae(tokens, patch_fn)
        diff = patched.logits - normal.logits
        if return_prob_diffs:
            return diff, (patched.logits.softmax(-1) - normal.logits.softmax(-1))
        return diff

    def detokenize(self, tokens):
        if tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)
        return self.llm.tokenizer._tokenizer.decode_batch(
            [[t] for t in tokens],
            skip_special_tokens=False,
        )

    def forward_token_attribution_to_features(self, tokens, seq_range):
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

        with fwAD.dual_level():
            with self.nnsight_model.trace(tokens) as tracer:
                embed = self.nnsight_model.transformer.wte.output
                self.nnsight_model.transformer.wte.output = nnsight.apply(
                    tangentize_embedding, embed
                )
                lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
                res = self.sae_with_patch(lambda x: x, return_sae_acts=True)(lm_acts)
                sae_acts = res[1]

                acts_tangent = nnsight.apply(fwAD.unpack_dual, sae_acts).tangent.save()
                lm_acts.stop()
        return acts_tangent

    def patching_effect_on_dataset(self, feature_id, batch_size=32, scale=0):
        """
        this will be like,
            - on the filtered dataset,
            - where the feature occurs
                - (for each position),
        what happens when we ablate the feature?
        """
        feature = self.features[feature_id]
        feature_active = feature.indices()
        feature = feature.to_dense()
        results = torch.zeros(128, 50257).cuda()
        p_results = torch.zeros(128, 50257).cuda()
        with torch.no_grad():
            for chunk in tqdm.tqdm(self.saved_acts.chunks):
                # print("new chunk")
                tokens = chunk.tokens
                docs, mask = tokens.index_where_valid(feature_active[0:1])
                seq_pos = feature_active[1, mask]
                for i in range(0, docs.shape[0], batch_size):
                    # print("did batch", i)
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

                    ldiff, pdiff = self.patchdiff(
                        batch_docs, patch_fn, return_prob_diffs=True
                    )

                    for j in range(batch_docs.shape[0]):
                        results[: -batch_seq_pos[j]] += ldiff[j, batch_seq_pos[j] :]
                        p_results[: -batch_seq_pos[j]] += pdiff[j, batch_seq_pos[j] :]
        return results, p_results

    def avg_fwad_effect_on_dataset(self, feature_id, batch_size=32, scale=1):
        """
        this will be like,
            - on the filtered dataset,
            - where the feature occurs
                - (for each position),
        what happens when we ablate the feature?
        """
        feature = self.features[feature_id]
        feature_active = feature.indices()
        feature = feature.to_dense()
        results = torch.zeros(128, 50257).cuda()
        p_results = torch.zeros(128, 50257).cuda()
        with torch.no_grad():
            for chunk in tqdm.tqdm(self.saved_acts.chunks):
                # print("new chunk")
                tokens = chunk.tokens
                docs, mask = tokens.index_where_valid(feature_active[0:1])
                seq_pos = feature_active[1, mask]
                for i in range(0, docs.shape[0], batch_size):
                    # print("did batch", i)
                    batch_docs = docs[i : i + batch_size]
                    batch_seq_pos = seq_pos[i : i + batch_size]

                    def tangent_gen(acts):
                        tangent = torch.zeros_like(acts)
                        tangent[
                            torch.arange(batch_seq_pos.shape[0]),
                            batch_seq_pos,
                            feature_id,
                        ] = 1
                        return acts

                    def patch_fn(acts):
                        acts = acts.clone()
                        acts[
                            torch.arange(batch_seq_pos.shape[0]),
                            batch_seq_pos,
                            feature_id,
                        ] *= scale
                        return acts

                    effect = self.forward_ad_with_sae(
                        batch_docs,
                        tangent_gen=tangent_gen,
                        patch_fn=patch_fn,
                        return_prob_grads=True,
                    )
                    # effect = out.logits.softmax(-1) * effect
                    # effect.tangent?
                    for j in range(batch_docs.shape[0]):
                        results[: -batch_seq_pos[j]] += effect.tangent[
                            j, batch_seq_pos[j].item() :
                        ]
        return results
