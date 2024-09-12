from .acts_cacher import ActsCacher, CachingConfig
from .saved_acts import SavedActs
from saeco.trainer import TrainingRunner, RunConfig
from attr import define, field
import torch
from pathlib import Path
from torch import Tensor
import tqdm
import einops
import nnsight
from .nnsite import getsite, setsite, tlsite_to_nnsite
import torch.autograd.forward_ad as fwAD


@define
class Evaluation:
    model_name: str
    training_runner: TrainingRunner = field(repr=False)
    cache_path: Path | None = None
    saved_acts: SavedActs | None = field(default=None, repr=False)
    cache_name: str | None = None
    nnsight_model: nnsight.LanguageModel | None = field(default=None, repr=False)

    def store_acts(self, caching_cfg: CachingConfig, displace_existing=False):
        if caching_cfg.model_name is None:
            caching_cfg.model_name = self.model_name
        assert caching_cfg.model_name == self.model_name
        acts_cacher = ActsCacher(caching_cfg, self.training_runner, self.cache_path)
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
        self.saved_acts = SavedActs(acts_cacher.path())

    @property
    def path(self):
        if self.cache_name is None:
            raise ValueError("cache_name must be set")
        return Path.home() / "workspace" / "cached_sae_acts" / self.cache_name

    @classmethod
    def from_cache_name(cls, name: Path | str):
        if isinstance(name, str):
            name = Path(name)
            if not name.exists():
                name = Path.home() / "workspace" / "cached_sae_acts" / name
        saved = SavedActs(name)
        inst = cls.from_model_name(saved.cfg.model_name)
        inst.saved_acts = saved
        return inst

    @classmethod
    def from_model_name(cls, name: str):
        tr = TrainingRunner.autoload(name)
        inst = cls(training_runner=tr, model_name=name)
        return inst

    # def get_features_and_active_docs(self, feature_ids):
    #     feature_tensors = [
    #         self.saved_acts.active_feature_tensor(fid) for fid in feature_ids
    #     ]
    #     f = feature_tensors[0].clone()
    #     for ft in feature_tensors[1:]:
    #         f += ft
    #     assert f.is_sparse
    #     f = f.coalesce()
    def get_active_documents(self, feature_ids):
        features = self.get_features(feature_ids)
        return self.select_active_documents(self.features_union(features))

    def get_features(self, feature_ids):
        return [self.saved_acts.active_feature_tensor(fid) for fid in feature_ids]

    def get_feature(self, feature_id):
        return self.saved_acts.active_feature_tensor(feature_id)

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
            size=(self.saved_acts.tokens.ndoc, docs.shape[1]),
        ).coalesce()

    def _active_docs_values_and_indices(self, feature):
        active_documents_idxs = self.active_document_indices(feature)
        active_documents = self.saved_acts.tokens[
            active_documents_idxs
        ]  # unsqueeze gets this back into the normal indexing shape
        return active_documents, active_documents_idxs

    def get_top_activating(self, feature: Tensor, percentile=10):
        topk = feature.values().topk(int(feature.values().shape[0] * percentile / 100))
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
            acts = chunk.acts.cuda().to_dense()
            fds = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            f2s = fds.pow(2).sum(-1)
            assert f2s.shape == (self.d_dict,)
            f2sum += f2s
            mat += fds @ fds.transpose(-2, -1)
        norms = f2sum.sqrt()
        mat /= norms.unsqueeze(0)
        mat /= norms.unsqueeze(1)
        prod = mat.diag().prod()
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
            acts = chunk.acts.cuda().to_dense()
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
        prod = mat.diag().prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    def co_occurrence(self, threshold):
        """
        Returns the masked cosine similarities matrix.
        Indexes are like: [masking feature, masked feature]
        """
        # threshold = 0
        # mat = torch.zeros(self.d_dict, self.d_dict).cuda()
        # f2sum = torch.zeros(self.d_dict).cuda()
        # maskedf2sum = torch.zeros(self.d_dict, self.d_dict).cuda()
        # for chunk in tqdm.tqdm(
        #     self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        # ):
        #     acts = chunk.acts.cuda().to_dense()
        #     feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
        #     feats_mask = feats_mat > threshold

        #     f2s = feats_mat.pow(2).sum(-1)
        #     assert f2s.shape == (self.d_dict,)
        #     f2sum += f2s
        #     maskedf2sum += feats_mask.float() @ feats_mat.transpose(-2, -1).pow(2)
        #     mat += feats_mat @ feats_mat.transpose(-2, -1)
        # norms = f2sum.sqrt()
        # mat /= maskedf2sum.sqrt()
        # mat /= norms.unsqueeze(1)
        # prod = mat.diag().prod()
        # assert prod < 1.001 and prod > 0.999
        # return mat

    def document_co_occurrence(self, threshold): ...

    @property
    def d_dict(self):
        return self.training_runner.cfg.init_cfg.d_dict

    @property
    def sae_cfg(self) -> RunConfig:
        return self.training_runner.cfg

    @property
    def cache_cfg(self) -> CachingConfig:
        return self.saved_acts.cfg

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

    def run_with_sae2(
        self,
        tokens,
        patch=lambda x: x,
        sae_acts_hook=lambda x: x,
        out_hook=lambda x: x,
    ):
        assert tokens.ndim == 2
        with self.nnsight_model.trace(tokens) as tracer:
            lm_acts = getsite(self.nnsight_model, self.nnsight_site_name)
            res = self.sae_with_patch(patch, return_sae_acts=True)(lm_acts)
            nnsight.apply(sae_acts_hook, res[1])
            setsite(self.nnsight_model, self.nnsight_site_name, res[0])
            out = self.nnsight_model.output
            hout = nnsight.apply(out_hook, out)
            out = hout if hout is not None else out
            out = out.save()
        return out

    @property
    def nnsight_site_name(self):
        return tlsite_to_nnsite(
            self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site
        )

    def forward_ad_with_sae(
        self, tokens, tangent=None, tangent_gen=None, patch_hook=lambda x: x
    ):
        assert tokens.ndim == 2
        assert tangent or tangent_gen
        if tangent:

            def fwad_hook(acts):
                acts = patch_hook(acts)
                return fwAD.make_dual(acts, tangent)

        else:

            def fwad_hook(acts):
                acts = patch_hook(acts)
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

        return out, tangent

    def _skip_bos_if_appropriate(self, lm_acts, reconstructed_acts):
        if self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first:
            return torch.cat([lm_acts[:, :1], reconstructed_acts[:, 1:]], dim=1)
        return reconstructed_acts

    def patchdiff(self, tokens, patch_fn):
        normal = self.run_with_sae(tokens)
        patched = self.run_with_sae(tokens, patch_fn)
        return patched.logits - normal.logits

    def detokenize(self, tokens):
        return self.llm.tokenizer._tokenizer.decode_batch(
            [[t] for t in tokens],
            skip_special_tokens=False,
        )
