import shelve
from functools import cached_property, wraps
from pathlib import Path
from typing import Generator, Iterable, Iterator, Tuple, Union

import einops
import nnsight
import torch
import torch.autograd.forward_ad as fwAD
import tqdm
from attr import define, field

from pydantic import BaseModel
from torch import Tensor

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from saeco.evaluation.MetadataBuilder import MetadataBuilder
from saeco.trainer import RunConfig, TrainingRunner
from ..misc.nnsite import getsite, setsite
from .cached_artifacts import CachedCalls
from .cacher2 import ActsCacher, CachingConfig

from .fastapi_models import (
    Feature,
    MetadataEnrichmentLabelResult,
    MetadataEnrichmentResponse,
    MetadataEnrichmentSortBy,
    TokenEnrichmentMode,
    TokenEnrichmentSortBy,
)
from .fastapi_models.families_draft import Family, FamilyLevel, GetFamiliesResponse

from .filtered import FilteredTensor
from .metadata import Artifacts, Filters, Metadatas
from .named_filter import NamedFilter
from .saved_acts import SavedActs
from .storage.chunk import Chunk


@define
class BMStorShelf:
    path: Path
    shelf: shelve.Shelf

    @classmethod
    def from_path(cls, path: Path):
        return cls(
            path=path,
            shelf=shelve.open(str(path / "shelf")),
        )

    def fnkey(self, func, args, kwargs):
        key = f"{func.__name__}__{args}__{kwargs}"
        vkey = f"{key}__version"
        return key, vkey

    def version(self, func):
        return getattr(func, "_version", None)

    def has(self, func, args, kwargs):
        key, vkey = self.fnkey(func, args, kwargs)
        version = self.version(func)
        return vkey in self.shelf and self.shelf[vkey] == version and key in self.shelf

    def get(self, func, args, kwargs):
        key, vkey = self.fnkey(func, args, kwargs)
        return self.shelf[key]

    def set(self, func, args, kwargs, value):
        key, vkey = self.fnkey(func, args, kwargs)
        self.shelf[key] = value
        self.shelf[vkey] = self.version(func)
        self.shelf.sync()


def cache_version(v):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(wrapper, "_version", v)
        return wrapper

    return decorator


class Test(BaseModel):
    a: int
    b: int


@define
class Evaluation:
    model_name: str
    training_runner: TrainingRunner = field(repr=False)
    saved_acts: SavedActs | None = field(default=None, repr=False)
    _filter: NamedFilter | None = field(default=None)
    tokenizer: PreTrainedTokenizerFast = field()
    _root: Union["Evaluation", None] = field(default=None, repr=False)

    @tokenizer.default
    def _tokenizer_default(self):
        return AutoTokenizer.from_pretrained(
            self.sae_cfg.train_cfg.data_cfg.model_cfg.model_name
        )

    @cached_property
    def bmstore(self):
        return BMStorShelf.from_path(self.path)

    @property
    def root(self):
        if self._root is None:
            return self
        return self._root

    @property
    def cuda(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def docs(self):
        return self.saved_acts.tokens

    @property
    def docstrs(self) -> "StrDocs":
        return StrDocs(self)

    @property
    def acts(self):
        return self.saved_acts.acts

    @property
    def nnsight_model(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.model

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

    def __attrs_post_init__(self):
        if self.saved_acts is not None:
            if self.saved_acts.data_filter is not self._filter:
                raise ValueError("Filter mismatch between Evaluation and storage")
        else:
            assert self._filter is None

    def _apply_filter(self, filter: NamedFilter | Tensor):
        if isinstance(filter, Tensor):
            filter = NamedFilter(filter=filter, filter_name=None)
        if self._filter is not None:
            raise ValueError(
                "Filter already set, create filtered from the root Evaluation"
            )
        return Evaluation(
            model_name=self.model_name,
            training_runner=self.training_runner,
            saved_acts=self.saved_acts.filtered(filter),
            filter=filter,
            _root=self,
        )

    def open_filtered(self, filter_name: str):
        return self._apply_filter(self.filters[filter_name])

    def _make_metadata_builder_iter(
        self, dtype, device, item_size=[]
    ) -> Generator[Chunk, FilteredTensor, Tensor]:
        assert self._filter is None
        new_tensor = torch.zeros(
            self.cache_cfg.num_docs, *item_size, dtype=dtype, device=device
        )

        for chunk in self.saved_acts.chunks:
            value = yield chunk
            yield
            assert isinstance(value, FilteredTensor | Tensor)
            if isinstance(value, Tensor):
                value = chunk._to_filtered(value)
            value.filter.writeat(new_tensor, value.value)
        return new_tensor

    def metadata_builder(self, dtype, device, item_size=[]) -> "MetadataBuilder":
        return MetadataBuilder(
            self.saved_acts.chunks,
            dtype=dtype,
            device=device,
            shape=[self.cache_cfg.num_docs, *item_size],
        )
        return TensorBuilder(self._make_metadata_builder_iter(dtype, device, item_size))

    @property
    def path(self):
        if self.saved_acts is None:
            raise ValueError("cache_name must be set")
        if self._filter is None:
            return self.cache_cfg.path
        return self._filter.filtered_dir(self.cache_cfg.path)

    @cached_property
    def cached_call(self) -> Union[CachedCalls, "Evaluation"]:
        return CachedCalls(self)

    @cached_property
    def metadatas(self) -> Metadatas:
        if self._filter is not None:
            raise NotImplementedError(
                "Getting metadatas from a filtered Evaluation is TODO and pending some design choices."
            )
        return Metadatas(self.path, self.cache_cfg)

    @property
    def _root_metadatas(self):
        if self._root is None:
            return self.metadatas
        return self._root.metadatas

    @cached_property
    def artifacts(self) -> Artifacts:
        return Artifacts(self.path, self.cache_cfg)

    @cached_property
    def filters(self) -> Filters:
        if self._filter is not None:
            raise ValueError(
                "Cannot access filters from a filtered evaluation. If this could be useful though, let me (Glen) know."
            )

        return Filters(self.path, self.cache_cfg)

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
    def features(self):
        return self.saved_acts.features

    @property
    def sae(self):
        return self.training_runner.trainable

    @property
    def nnsight_site_name(self):
        return self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.site

    @property
    def seq_len(self):
        return self.sae_cfg.train_cfg.data_cfg.seq_len

    @property
    def d_vocab(self):
        return self.tokenizer.vocab_size

    def store_acts(self, caching_cfg: CachingConfig, displace_existing=False):
        if caching_cfg.model_name is None:
            caching_cfg.model_name = self.model_name
        assert caching_cfg.model_name == self.model_name
        acts_cacher = ActsCacher.from_cache_and_runner(
            caching_config=caching_cfg, model_context=self.training_runner
        )
        if acts_cacher.path.exists():
            if displace_existing:
                import time

                old = acts_cacher.path.parent / "old"
                old.mkdir(exist_ok=True, parents=True)
                acts_cacher.path.rename(
                    old / f"old_{time.time()}{acts_cacher.path.name}"
                )
            else:
                raise FileExistsError(
                    f"{acts_cacher.path} already exists. Set displace_existing=True to move existing files."
                )

        metadata_chunks = acts_cacher.store_acts()
        self.saved_acts = SavedActs.from_path(acts_cacher.path)
        metadata_builders = {
            name: self.metadata_builder(torch.long, "cpu")
            for name in self.cache_cfg.metadatas_from_src_column_names
        }
        for mchunk in metadata_chunks:
            for name in mchunk:
                metadata_builders[name].takestrl(mchunk[name])
        for name, builder in metadata_builders.items():
            self.metadatas[name] = builder.value
            self.metadatas.set_str_translator(name, builder.unique_labels)

    def get_features(self, feature_ids):
        return [self.features[fid] for fid in feature_ids]

    def get_feature(self, feature_id) -> FilteredTensor:
        return self.features[feature_id]

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

    def activation_cosims(self):
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
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
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        maskedf2sum = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
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

    def doc_level_co_occurrence(self, pooling="mean"):
        """
        Pooling: "mean", "max" or "binary"
        this could be done at sequence level if we want
        """
        threshold = 0
        mat = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        f2sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
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

    def sequelize(
        self,
        acts: Tensor,
        doc_agg: float | int | str | None = None,
    ):
        assert acts.ndim == 3
        if doc_agg:
            if isinstance(doc_agg, float | int):
                acts = acts.pow(doc_agg).sum(dim=1).pow(1 / doc_agg)
            elif doc_agg == "count":
                acts = (acts > 0).sum(dim=1).float()
            elif doc_agg == "max":
                acts = acts.max(dim=1).values
            else:
                raise ValueError("Invalid doc_agg")
            return einops.rearrange(acts, "doc feat -> feat doc")
        else:
            return einops.rearrange(acts, "doc seq feat -> feat (doc seq)")

    def coactivations(self, doc_agg=None):
        sims = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        coact_counts = torch.zeros(self.d_dict, self.d_dict).to(self.cuda)
        fa_sq_sum = torch.zeros(self.d_dict).to(self.cuda)
        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            assert acts.ndim == 3
            feature_activity = self.sequelize(
                acts, doc_agg=doc_agg
            )  # feat, (doc [seq])
            feature_bin = (feature_activity > 0).float()
            fa_sq_sum += feature_activity.pow(2).sum(-1)
            sims += feature_activity @ feature_activity.transpose(-2, -1)
            coact_counts += feature_bin @ feature_bin.transpose(-2, -1)
        norms = fa_sq_sum.sqrt()
        sims /= norms.unsqueeze(0)
        sims /= norms.unsqueeze(1)
        prod = sims.diag()[~sims.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return coact_counts, sims

    def cosims(self, doc_agg=None):
        return self.coactivations(doc_agg=doc_agg)[1]

    def coactivity(self, doc_agg=None):
        res = self.coactivations(doc_agg=doc_agg)
        self.artifacts[f"cosims({(doc_agg,)}, {{}})"] = res[1]
        return res[0]

    def generate_feature_families1(
        self, doc_agg=None, threshold=0.1, n=3, use_D=False, freq_bounds=None
    ):
        # C_unnormalized, D = self.coactivations(doc_agg=doc_agg)
        if use_D:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        else:
            unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        # D = D.cpu()
        C = unnormalized / (
            (
                feat_counts := (
                    self.doc_activation_counts
                    if doc_agg
                    else self.seq_activation_counts
                )
            )
            .cpu()
            .unsqueeze(-1)
            + 1e-6
        )
        threshold = threshold or C[C > 0].median()

        C[C.isnan()] = 0
        C[C < threshold] = 0
        if freq_bounds is not None:
            fmin, fmax = freq_bounds
            feat_probs = (
                self.doc_activation_probs if doc_agg else self.seq_activation_probs
            )
            bound = (feat_probs >= fmin) & (feat_probs <= fmax)
            C[~bound] = 0
            C[:, ~bound] = 0
        import scipy.sparse as ssp

        from .mst import Families, FamilyTreeNode, mst, my_mst

        levels = []
        feat_counts = feat_counts.to(self.cuda)
        for _ in tqdm.trange(n):
            tree = mst(C).transpose(0, 1)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            # for i in range(nz.shape[0]):
            #     c = nz[i]
            #     assert feat_counts[c[0]] >= feat_counts[c[1]]
            # families = Families.from_tree(tree)
            levels.append(tree)
            C[roots] = 0
            C[:, roots] = 0
        # roots = [((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0) for tree in levels]

        return levels

    @torch.no_grad()
    def generate_feature_families(self, doc_agg=None, threshold=None, n=3, use_D=False):
        if use_D:
            unnormalized = self.cached_call.cosims(doc_agg=doc_agg).cpu()
        else:
            unnormalized = self.cached_call.coactivity(doc_agg=doc_agg).cpu()
        unnormalized[unnormalized.isnan()] = 0
        feat_counts = (
            self.doc_activation_counts if doc_agg else self.seq_activation_counts
        )
        # feat_probs = self.doc_activation_probs if doc_agg else self.seq_activation_probs

        def denan(x):
            return torch.where(x.isnan() | x.isinf(), torch.zeros_like(x), x)

        def zdiag(x):
            return torch.where(
                torch.eye(x.shape[0], device=x.device, dtype=torch.bool), 0, x
            )

        def isprob(P):
            assert (P >= 0).all() and (P <= 1).all()
            return P

        def probmat(P):
            return isprob(zdiag(denan(P)))

        def ent(P):
            P = isprob(denan(P))
            return torch.where(
                (P > 0) & (P < 1),
                -P * torch.log(P + 1e-6) - (1 - P) * torch.log(1 - P + 1e-6),
                0,
            )

        def nicemat(M):
            return zdiag(denan(M))

        def nent(Q, R):
            Q, R = nicemat(Q), nicemat(R)
            P = nicemat(Q / R)
            isprob(P)
            return torch.where(
                (P > 0) & (P < 1),
                -Q * torch.log(P) - (R - Q) * torch.log(1 - P),
                0,
            )

        N = self.num_docs if doc_agg else self.seq_len * self.num_docs
        A = feat_counts.unsqueeze(1).expand(-1, feat_counts.shape[0])
        B = feat_counts.unsqueeze(0).expand(feat_counts.shape[0], -1)
        V = unnormalized

        P_A = probmat(A / N)
        P_B = probmat(B / N)
        # P_AB = probmat(V / N)
        P_B_given_A = probmat((V + P_B * P_A) / (A + 1))  # +
        P_B_given_not_A = probmat((B - V + (1 - P_B) * (1 - P_A)) / (N - A + 1))
        # C = P_A * torch.log()
        # info = A * ent(P_B_given_A) + (N - A) * ent(P_B_given_not_A)
        # info = P_A * nent(V, A) + (1 - P_A) * nent((B - V), (N - A)) - nent(B, N)
        # info = (c := ent(P_B)) - (
        #     (a := P_A * ent(P_B_given_A)) + (b := (1 - P_A) * ent(P_B_given_not_A))
        # )
        # info = torch.where((P_B_given_A > P_B), info + 1e-6, 0)
        # (info[P_B_given_A > P_B]).sum()
        # (P_B_given_A > P_B).sum() / (P_B_given_A < P_B).sum()
        # r = P_A * torch.log(P_B_given_A / P_B)
        r = torch.log(P_B_given_A / P_B)
        r = ent(P_B) - ent(P_B_given_A)

        t = (1 - P_A) * torch.log(P_B_given_not_A / P_B)
        # other = P_A * torch.log(P_B_given_A / P_B) + (1 - P_A) * torch.log(
        #     P_B_given_not_A / P_B
        # )

        r = denan(r)
        t = denan(t)
        # r[P_B_given_A > P_B].sum()
        # r[P_B_given_A < P_B].sum()
        # t[P_B_given_A > P_B].sum()
        # t[P_B_given_A < P_B].sum()
        info = r

        # a.max(dim=0, keepdim=True)
        # b.max(dim=0, keepdim=True)
        # c.max(dim=0, keepdim=True)
        # a.max(dim=1, keepdim=True)
        # b.max(dim=1, keepdim=True)
        # c.max(dim=1, keepdim=True)
        # P_B.max()
        # v = B.to(torch.float64) / N
        # v.max()

        info = torch.where((V > 0) & (info > 0), info, 0)

        info = zdiag(denan(info))
        assert (info >= 0).all()
        # learned =
        # C = info.clone()
        C = info
        threshold = threshold or C[C > 0].median()
        # C = unnormalized / (().cpu().unsqueeze(-1) + 1e-6)

        C.max(dim=0, keepdim=True)
        C.max(dim=1, keepdim=True)

        # C[C.isnan()] = 0
        C[C < threshold] = 0
        import scipy.sparse as ssp

        from .mst import Families, FamilyTreeNode, mst, my_mst, prim_max

        levels = []
        feat_counts = feat_counts.to(self.cuda)
        for _ in range(n):
            # tree = mst(C)
            i, v = my_mst(C.cuda())
            tree = (
                torch.sparse_coo_tensor(indices=i, values=v, size=C.shape)
                .to_dense()
                .cpu()
                .transpose(0, 1)
            )
            nz = tree.nonzero()
            print(
                "proportion:",
                (feat_counts[nz[:, 0]] >= feat_counts[nz[:, 1]]).float().mean(),
            )

            # families = Families.from_tree(tree)
            roots = ((tree > 0).sum(dim=0) == 0) & ((tree > 0).sum(dim=1) > 0)
            print("num_roots", roots.sum())
            levels.append(tree)
            C[roots] = 0
            C[:, roots] = 0
        # roots.sum()

        # (
        #     feat_counts[levels[0].roots[0].feature_id],
        #     feat_counts[levels[1].roots[0].feature_id],
        #     feat_counts[levels[2].roots[0].feature_id],
        # )
        # (len(levels[0].roots[0]), len(levels[1].roots[0]), len(levels[2].roots[0]))

        # (len(levels[0]), len(levels[1]), len(levels[2]))

        return levels

    @torch.no_grad()
    def _get_feature_family_trees(
        self, doc_agg=None, threshold=None, n=3, use_D=False, freq_bounds=None
    ):
        return torch.stack(
            self.generate_feature_families1(
                doc_agg=doc_agg,
                threshold=threshold,
                n=n,
                use_D=use_D,
                freq_bounds=freq_bounds,
            )
        )

    def get_feature_families(self) -> GetFamiliesResponse:
        from .mst import Families, FamilyTreeNode

        levels = self.cached_call._get_feature_family_trees()
        famlevels = [[Families.from_tree(f) for f in fam] for fam in levels]

        niceroots: list[list[FamilyTreeNode]] = [
            [r for r in f.roots if len(r) > 10] for f in famlevels
        ]
        levels = []
        for levelnum, level in enumerate(niceroots):
            fl = FamilyLevel(
                level=levelnum,
                families={
                    root.feature_id: Family(
                        level=levelnum,
                        family_id=fam_id,
                        label=None,
                        subfamilies=[],
                        subfeatures=[
                            (Feature(feature_id=int(feat_id), label=None), 0.9)
                            for feat_id in root.family
                        ],
                    )
                    for fam_id, root in enumerate(level)
                },
            )
            levels.append(fl)
        level_lens = [len(l.families) for l in levels]
        maxl = sum(level_lens)
        csll = torch.tensor(level_lens).cumsum(0).tolist()
        level_tensors = []
        t0 = torch.zeros(
            len(levels), maxl, self.d_dict, dtype=torch.bool, device=self.cuda
        )
        for i, level in enumerate(levels):
            for j, family in level.families.items():
                for feat in family.subfeatures:
                    t0[csll[i] + j, feat.feature_id] = True

    def top_coactivating_features(self, feature_id, top_n=10, mode="seq"):
        """
        mode: "seq" or "doc"
        """
        if mode == "seq":
            mat = self.cached_call.activation_cosims()
        elif mode == "doc":
            mat = self.cached_call.doc_level_co_occurrence()
        else:
            raise ValueError("mode must be 'seq' or 'doc'")
        vals = mat[feature_id]
        vals[feature_id] = -torch.inf
        vals[vals.isnan()] = -torch.inf
        top = vals.topk(top_n + 1)

        v = top.values
        i = top.indices
        v = v[i != feature_id][:top_n]
        i = i[i != feature_id][:top_n]
        return i, v

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
                ls_logits = out.logits.log_softmax(dim=-1)
                tangent = nnsight.apply(fwAD.unpack_dual, ls_logits).tangent.save()
            if return_prob_grads:
                soft = out.logits.softmax(dim=-1)
                probspace = fwAD.unpack_dual(soft).tangent
        if return_prob_grads:
            return out, tangent, probspace
        return out, tangent

    def _skip_bos_if_appropriate(self, lm_acts, reconstructed_acts):
        if self.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.excl_first:
            return torch.cat([lm_acts[:, :1], reconstructed_acts[:, 1:]], dim=1)
        return reconstructed_acts

    def detokenize(self, tokens) -> list[str] | list[list[str]] | str:
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        assert isinstance(
            tokens, Tensor
        ), "hmu if this assumption is wrong somewhere, easy fix"
        if tokens.ndim == 0:
            return self.tokenizer._tokenizer.decode([tokens])
        if tokens.ndim == 1:
            return self.tokenizer._tokenizer.decode_batch(
                [[t] for t in tokens],
                skip_special_tokens=False,
            )

        lens = tokens.shape[1]
        flat = einops.rearrange(tokens, "doc seq -> (doc seq)").unsqueeze(-1).tolist()
        flatl = self.tokenizer._tokenizer.decode_batch(flat, skip_special_tokens=False)
        return [flatl[i : i + lens] for i in range(0, len(flatl), lens)]

    def patchdiff(self, tokens, patch_fn, return_prob_diffs=False, invert=False):
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

    def seq_aggregated_chunks_yielder(
        self, seq_agg
    ) -> Generator[FilteredTensor, None, None]:
        """
        seq_agg options: "mean", "max", "sum", "count", "any"
        - count: count number of non-zero activations in each doc
        - any: if feature has any non-zero activation in a doc

        """
        # move to saved acts?
        for chunk in tqdm.tqdm(self.saved_acts.chunks):
            acts = chunk.acts
            acts_inner = acts.value.to(self.cuda).to_dense()
            assert acts_inner.ndim == 3
            if seq_agg == "count":
                c_agg = (acts_inner > 0).sum(dim=1)
            elif seq_agg == "any":
                c_agg = (acts_inner > 0).any(dim=1)
            elif seq_agg == "max":
                c_agg = acts_inner.max(dim=1).values
            else:
                c_agg = getattr(acts_inner, seq_agg)(dim=1)
            yield acts.to_filtered_like_self(c_agg)

    @property
    def num_docs(self):
        if self._filter:
            return self._filter.filter.sum()
        return self.cache_cfg.num_docs

    def acts_avg_over_dataset(self, seq_agg="mean", docs_agg="mean"):
        """
        seq_agg options: "mean", "max", "sum", "count", "any"
        docs_agg options: "mean", "max", "sum"
        """
        results = torch.zeros(self.d_dict).to(self.cuda)

        for agg_chunk in self.seq_aggregated_chunks_yielder(seq_agg):
            if docs_agg == "max":
                results = (
                    torch.cat([results, agg_chunk.value.max(dim=0).values])
                    .max(dim=0)
                    .values
                )
            else:
                results += agg_chunk.value.sum(dim=0)
        if docs_agg == "mean":
            results /= self.num_docs

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

    def average_patching_effect_on_dataset(
        self, feature_id, batch_size=8, scale=None, by_fwad=False, random_subset_n=None
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
        p_results = torch.zeros(
            self.seq_len,
            self.d_vocab,
        ).to(self.cuda)
        num_batches = torch.zeros(self.seq_len).to(self.cuda) + 1e-6
        for ldiff, pdiff, batch_seq_pos in self.patching_effect_on_dataset(
            feature_id=feature_id,
            batch_size=batch_size,
            scale=scale,
            by_fwad=by_fwad,
            random_subset_n=random_subset_n,
        ):
            for j in range(batch_seq_pos.shape[0]):
                results[: -batch_seq_pos[j]] += ldiff[j, batch_seq_pos[j] :]
                p_results[: -batch_seq_pos[j]] += pdiff[j, batch_seq_pos[j] :]
                num_batches[: -batch_seq_pos[j]] += 1
        return results / num_batches.unsqueeze(-1), p_results / num_batches.unsqueeze(
            -1
        )

    def custom_patching_effect_aggregation(
        self,
        feature_id,
        log_call,
        prob_call,
        batch_size=8,
        scale=None,
        by_fwad=False,
        random_subset_n=None,
    ):
        num_batches = torch.zeros(self.seq_len) + 1e-6
        for ldiff, pdiff, batch_seq_pos in self.patching_effect_on_dataset(
            feature_id=feature_id,
            batch_size=batch_size,
            scale=scale,
            by_fwad=by_fwad,
            random_subset_n=random_subset_n,
        ):
            if log_call is not None:
                log_call(ldiff, batch_seq_pos)
            if prob_call is not None:
                prob_call(pdiff, batch_seq_pos)
        return num_batches

    def patching_effect_on_dataset(
        self, feature_id, batch_size=8, scale=None, by_fwad=False, random_subset_n=None
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
        feature0 = self.features[feature_id]
        feature = feature0.filter_inactive_docs()
        if random_subset_n:
            if (s := feature.filter.mask.sum()) > random_subset_n:
                new_mask = torch.zeros_like(feature.filter.mask)
                new_mask[feature.filter.mask] = torch.randperm(s) < random_subset_n
                assert new_mask.sum() == random_subset_n
                new_feature = feature.mask_by_other(
                    new_mask, return_ft=True, presliced=True, value_like=False
                )
                nfi = new_feature.indices()
                fi = feature.indices().transpose(0, 1).tolist()
                if nfi.numel() > 0:
                    for i in nfi.transpose(0, 1).tolist():
                        assert i in fi
                feature = new_feature
        feature_active = feature.indices()
        feature = feature.to_dense()

        with torch.no_grad():
            for chunk in tqdm.tqdm(self.saved_acts.chunks):
                tokens = chunk.tokens
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
                            return acts

                        out, ldiff, pdiff = self.forward_ad_with_sae(
                            batch_docs,
                            tangent_gen=tangent_gen,
                            patch_fn=patch_fn,
                            return_prob_grads=True,
                        )
                        # ldiff = ldiff.log_softmax(-1
                        # not clear on if these should get some sort of
                        # normalization to prevent overweighting of
                        # particular documents w systematically larger grads
                        # (eg due to diff logit scales)
                        # maybe like z-score it or something
                        # unclear if log-softmax would be a coherent thing to do
                        # oh i think log-softmax on the dual tensor inside forward-ad
                        # should do what we want.
                        # seems good, added to forward_ad_with_sae

                    else:
                        ldiff, pdiff = self.patchdiff(
                            batch_docs, patch_fn, return_prob_diffs=True, invert=True
                        )
                    yield ldiff, pdiff, batch_seq_pos

    @property
    def token_occurrence_count(self):
        return self.cached_call._get_token_occurrences()

    def _get_token_occurrences(self):
        counts = torch.zeros(self.d_vocab, dtype=torch.long).to(self.cuda)
        for chunk in tqdm.tqdm(self.saved_acts.chunks):
            tokens = chunk.tokens.value.to(self.cuda)
            t, c_counts = tokens.unique(return_counts=True)
            counts[t] += c_counts
        return counts

    def top_activating_examples(self, feature_id: int, p=None, k=None):
        feature = self.features[feature_id]
        top = self._get_top_activating(feature.value, p=p, k=k)
        return feature.to_filtered_like_self(top)

    @staticmethod
    def _pk_to_k(p, k, quantity):
        if (p is None) == (k is None):
            raise ValueError("Exactly one of p and k must be set")
        if p is not None and not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        if k is None:
            k = int(quantity * p)
        if k <= 0:
            raise ValueError("k must be positive")
        return k

    @staticmethod
    def _get_top_activating(feature: Tensor, p=None, k=None):
        k = Evaluation._pk_to_k(p, k, feature.shape[0])
        values = feature.values()
        if k >= values.shape[0]:
            k = values.shape[0]
        topk = values.topk(k)

        return torch.sparse_coo_tensor(
            feature.indices()[:, topk.indices],
            topk.values,
            feature.shape,
        )

    def seq_agg_feat(self, feature_id=None, feature=None, agg="max", docs_filter=True):
        assert agg == "max", "Only max implemented currently"
        if (feature_id is None) == (feature is None):
            raise ValueError("Exactly one of feat_id and feature must be set")
        if feature is None:
            feature = self.features[feature_id]
        if docs_filter:
            feature = feature.filter_inactive_docs()
        return feature.to_filtered_like_self(
            feature.value.to_dense().max(dim=1).values, ndim=1
        )

    def top_activations_and_metadatas(
        self,
        feature: int | FilteredTensor,
        p: float = None,
        k: int = None,
        metadata_keys: list[str] = [],
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
    ):
        if isinstance(feature, int):
            feature = self.features[feature]
        doc_acts = self.seq_agg_feat(feature=feature)
        k = Evaluation._pk_to_k(p, k, doc_acts.value.shape[0])
        topk = doc_acts.value.topk(k, sorted=True)
        top_outer_indices = doc_acts.externalize_indices(topk.indices.unsqueeze(0))
        acts = feature.index_select(top_outer_indices[0], dim=0)
        assert (acts.to_dense() == feature.to_dense()[top_outer_indices]).all()
        if return_acts_sparse:
            acts = acts.to_sparse_coo()
        doc_indices = top_outer_indices[0]
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]
        metadatas = {
            key: self._root_metadatas[key][doc_indices] for key in metadata_keys
        }
        if str_metadatas:
            metadatas = self._root_metadatas.translate(metadatas)
        if return_doc_indices:
            return docs, acts, metadatas, doc_indices
        return docs, acts, metadatas

    def _metadata_unique_labels_and_counts_tensor(self, key):
        meta = self._root_metadatas[key]
        if self._filter is not None:
            meta = meta[self._filter.filter]
        assert meta.ndim == 1 and meta.dtype == torch.long
        labels, counts = meta.unique(return_counts=True)
        return torch.stack([labels, counts], dim=0)

    def top_activations_metadata_enrichments(
        self,
        *,
        feature: int | FilteredTensor,
        metadata_keys: list[str],
        p: float = None,
        k: int = None,
        str_label: bool = False,
        sort_by: MetadataEnrichmentSortBy = MetadataEnrichmentSortBy.counts,
    ):
        docs, acts, metadatas, doc_ids = self.top_activations_and_metadatas(
            feature=feature, p=p, k=k, metadata_keys=metadata_keys
        )
        r = {}
        for mdname, md in metadatas.items():
            assert md.ndim == 1
            full_lc = self.cached_call._metadata_unique_labels_and_counts_tensor(mdname)
            labels, mdcat_counts = torch.cat([md, full_lc[0]]).unique(
                return_counts=True
            )
            counts = mdcat_counts - 1
            assert (labels == full_lc[0]).all()
            assert counts.shape == labels.shape == full_lc[1].shape
            proportions = counts / full_lc[1]
            labels = labels[counts > 0]
            proportions = proportions[counts > 0]
            counts = counts[counts > 0]
            normalized_counts = proportions * self.num_docs / doc_ids.shape[0]
            scores = normalized_counts.log()
            if sort_by == TokenEnrichmentSortBy.counts:
                i = counts.argsort(descending=True)
            elif sort_by == TokenEnrichmentSortBy.normalized_count:
                i = normalized_counts.argsort(descending=True)
            elif sort_by == TokenEnrichmentSortBy.score:
                i = scores.argsort(descending=True)
            else:
                raise ValueError(f"Unknown sort_by {sort_by}")
            labels = labels[i]
            counts = counts[i]
            proportions = proportions[i]
            normalized_counts = normalized_counts[i]
            scores = scores[i]
            r[mdname] = [
                MetadataEnrichmentLabelResult(
                    label=label,
                    count=count,
                    proportion=proportion,
                    normalized_count=normalized_count,
                    score=score,
                    # **(dict(act_sum=acts[md == label].sum()) if return_act_sum else {}),
                )
                for label, count, proportion, normalized_count, score in zip(
                    (
                        self.metadatas.get(mdname).strlist(labels)
                        if str_label
                        else labels.tolist()
                    ),
                    counts.tolist(),
                    proportions.tolist(),
                    normalized_counts.tolist(),
                    scores.tolist(),
                )
            ]
        return MetadataEnrichmentResponse(results=r)

    def count_token_occurrence(self):
        counts = torch.zeros(self.d_vocab, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            toks = chunk.tokens.value.to(self.cuda).flatten()
            counts.scatter_add_(
                0,
                toks,
                torch.ones(1, device=toks.device, dtype=torch.long).expand(
                    toks.shape[0]
                ),
            )
        return counts

    def top_activations_token_enrichments(
        self,
        *,
        feature: int | FilteredTensor,
        p: float = None,
        k: int = None,
        mode: TokenEnrichmentMode = "doc",
        sort_by: TokenEnrichmentSortBy = "count",
    ):
        docs, acts, metadatas, doc_ids = self.top_activations_and_metadatas(
            feature=feature, p=p, k=k, metadata_keys=[]
        )
        docs = docs.to(self.cuda)
        if mode == TokenEnrichmentMode.doc:
            seltoks = docs
        elif mode == TokenEnrichmentMode.max:
            max_pos = acts.argmax(dim=1)
            max_top = docs[torch.arange(max_pos.shape[0]), max_pos]
            seltoks = max_top
        elif mode == TokenEnrichmentMode.active:
            active_top = docs[acts > 0]
            seltoks = active_top
        elif mode == TokenEnrichmentMode.top:
            top_threshold = docs[
                acts > acts.max(dim=-1).values.min(dim=0).values.item()
            ]
            seltoks = top_threshold
        else:
            raise ValueError(f"Unknown mode {mode}")
        tokens, counts = seltoks.flatten().unique(return_counts=True, sorted=True)
        normalized_counts = (counts / seltoks.numel()) / (
            self.token_occurrence_count.to(self.cuda)[tokens]
            / (self.num_docs * self.seq_len)
        )
        scores = normalized_counts.log()
        if sort_by == TokenEnrichmentSortBy.counts:
            i = counts.argsort(descending=True)
        elif sort_by == TokenEnrichmentSortBy.normalized_count:
            i = normalized_counts.argsort(descending=True)
        elif sort_by == TokenEnrichmentSortBy.score:
            i = scores.argsort(descending=True)
        else:
            raise ValueError(f"Unknown sort_by {sort_by}")
        tokens = tokens[i]
        counts = counts[i]
        normalized_counts = normalized_counts[i]
        scores = scores[i]

        return tokens, counts, normalized_counts, scores

    def num_active_docs_for_feature(self, feature_id):
        return self.cached_call._feature_num_active_docs()[feature_id].item()

    @property
    def seq_activation_counts(self):
        return self.cached_call._feature_num_active_tokens()

    @property
    def seq_activation_probs(self):
        return self.seq_activation_counts / (self.num_docs * self.seq_len)

    @property
    def doc_activation_counts(self):
        return self.cached_call._feature_num_active_docs()

    @property
    def doc_activation_probs(self):
        return self.doc_activation_counts / self.num_docs

    # @property
    # def mean_feature_activations(self)

    # def _feature_mean_activations(self):

    # def feature_activation_proportion_thresholds(self, p):

    def _feature_num_active_docs(self):
        activity = torch.zeros(self.d_dict, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).any(dim=1).sum(dim=0)
        return activity

    def _feature_num_active_tokens(self):
        activity = torch.zeros(self.d_dict, dtype=torch.long).to(self.cuda)
        for chunk in self.saved_acts.chunks:
            acts = chunk.acts.value.to(self.cuda).to_dense()
            activity += (acts > 0).sum(dim=1).sum(dim=0)
        return activity


@define
class StrDocs:
    eval: Evaluation

    def __getitem__(self, idx):
        tokens = self.eval.docs[idx]
        strs = self.eval.detokenize(tokens)
        assert len(strs) == tokens.shape[0] and len(strs[0]) == tokens.shape[1]
        return strs
