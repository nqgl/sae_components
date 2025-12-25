from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import tqdm
from torch import Tensor

from saeco.evaluation.fastapi_models.families_draft import (
    Family,
)
from saeco.evaluation.filtered import FilteredTensor
from saeco.evaluation.return_objects import Feature

if TYPE_CHECKING:
    from ..evaluation import Evaluation


class FamilyOps:
    def get_families_activations_on_docs(
        self: "Evaluation",
        families: list[Family],
        doc_indices: list[int],
        features: list[int] | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        str_metadatas: bool = False,
    ):
        if features is None:
            features = []
        if metadata_keys is None:
            metadata_keys = []
        doc_indices = torch.tensor(doc_indices, dtype=torch.long, device=self.cuda)
        print("getting families")
        print(self.cuda)
        docs, acts, metadatas = self.get_docs_acts_metadatas(
            doc_indices,
            features=self.get_family_psuedofeature_tensors(families=families)
            + [self.features[f].to(self.cuda) for f in features],
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            str_metadatas=str_metadatas,
        )
        return docs, acts[: len(families)], metadatas, acts[len(families) :]

    def top_activations_and_metadatas_for_family(
        self: "Evaluation",
        family: Family,
        aggregation_method: str = "sum",
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
    ):
        if metadata_keys is None:
            metadata_keys = []
        feature = self.get_family_psuedofeature_tensors([family], aggregation_method)[0]
        return self.top_activations_and_metadatas(
            feature=feature,
            p=p,
            k=k,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_acts_sparse=return_acts_sparse,
            return_doc_indices=return_doc_indices,
            str_metadatas=str_metadatas,
        )

    def _get_family_psuedofeature_artifact_names(
        self: "Evaluation", families: list[Family], aggregation_method: str
    ) -> list[str]:
        return [
            f"family-feature-tensor-{aggregation_method}_level{family.level}_family{family.family_id}_version{self._get_feature_families_unlabeled._version}"
            for family in families
        ]

    def get_family_psuedofeature_tensors(
        self: "Evaluation", families: list[Family], aggregation_method="sum", cuda=True
    ) -> list[FilteredTensor]:
        artifact_names = self._get_family_psuedofeature_artifact_names(
            families, aggregation_method
        )
        self.init_family_psuedofeature_tensors(families, aggregation_method)
        return [
            FilteredTensor.from_value_and_mask(
                (
                    self.artifacts[artifact_name].to(self.cuda)
                    if cuda
                    else self.artifacts[artifact_name]
                ),
                self.filter,
            )
            for artifact_name in artifact_names
        ]

    def init_family_psuedofeature_tensors(
        self: "Evaluation", families: list[Family], aggregation_method="sum"
    ) -> list[FilteredTensor]:
        artifact_names = self._get_family_psuedofeature_artifact_names(
            families, aggregation_method
        )
        precached = [
            artifact_name in self.artifacts for artifact_name in artifact_names
        ]

        if not all(precached):
            indices = [
                torch.tensor(
                    [f.feature.feature_id for f in family.subfeatures],
                    dtype=torch.long,
                    device=self.cuda,
                )
                for family, prec in zip(families, precached)
                if not prec
            ]
            new_artifact_names = [
                artifact_name
                for artifact_name, prec in zip(artifact_names, precached)
                if not prec
            ]
            builders = [
                self.filtered_builder(
                    dtype=torch.float, device=self.cuda, item_size=(self.seq_len,)
                )
                for _ in new_artifact_names
            ]
            for chunk in tqdm.tqdm(builders[0], total=self.cache_cfg.num_chunks):
                a = chunk.acts.to(self.cuda).to_dense()
                for mb, i in zip(builders, indices):
                    mb << a.to_filtered_like_self(a.value[:, :, i].sum(dim=-1), ndim=2)
            for artifact_name, mb in zip(new_artifact_names, builders):
                feature_value = mb.value
                self.artifacts[artifact_name] = feature_value.value

    def batched_top_activations_and_metadatas(
        self: "Evaluation",
        features: list[int | FilteredTensor],
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
    ):
        """
        Get top activations and metadatas for multiple features.

        When parallel=True (default), this uses efficient parallel processing
        to fetch top activations for all features concurrently.

        Args:
            features: List of feature IDs or FilteredTensor objects
            p: Proportion of top activations (alternative to k)
            k: Number of top activations
            metadata_keys: List of metadata keys to fetch
            return_str_docs: Whether to return string documents
            return_acts_sparse: Whether to return sparse activations
            return_doc_indices: Whether to return document indices
            str_metadatas: Whether to return string metadatas
            parallel: Whether to use parallel processing (default: True)
            max_workers: Maximum number of worker threads for parallel mode

        Returns:
            List of tuples (docs, acts, metadatas, [doc_indices])
        """
        if metadata_keys is None:
            metadata_keys = []

        if not features:
            return []

        if not parallel:
            # Sequential fallback
            return [
                self.top_activations_and_metadatas(
                    feature,
                    p,
                    k,
                    metadata_keys,
                    return_str_docs,
                    return_acts_sparse,
                    return_doc_indices,
                    str_metadatas,
                )
                for feature in features
            ]

        # Parallel processing: batch compute TopActivations first
        feature_objs = Feature.make_batch(src_eval=self, features=features)
        top_activations_list = Feature.batched_top_activations(
            features=feature_objs,
            p=p,
            k=k,
        )

        # Then extract legacy format from each TopActivations
        # Use thread pool for concurrent metadata fetching (I/O bound)
        def process_single(top_acts):
            return self._legacy_top_activations_and_metadatas_getter(
                top_acts=top_acts,
                metadata_keys=metadata_keys,
                return_str_docs=return_str_docs,
                return_doc_indices=return_doc_indices,
                str_metadatas=str_metadatas,
            )

        if len(top_activations_list) > 1 and max_workers != 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_single, top_activations_list))
        else:
            results = [process_single(ta) for ta in top_activations_list]

        return results

    def batched_top_activations_and_metadatas_for_family(
        self: "Evaluation",
        families: list[Family],
        aggregation_method: str = "sum",
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        return_acts_sparse: bool = False,
        return_doc_indices: bool = True,
        str_metadatas: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
    ):
        """
        Get top activations and metadatas for multiple families.

        When parallel=True (default), uses efficient parallel processing.

        Args:
            families: List of Family objects
            aggregation_method: Method for aggregating family features ("sum", etc.)
            p: Proportion of top activations (alternative to k)
            k: Number of top activations
            metadata_keys: List of metadata keys to fetch
            return_str_docs: Whether to return string documents
            return_acts_sparse: Whether to return sparse activations
            return_doc_indices: Whether to return document indices
            str_metadatas: Whether to return string metadatas
            parallel: Whether to use parallel processing (default: True)
            max_workers: Maximum number of worker threads for parallel mode

        Returns:
            List of tuples (docs, acts, metadatas, [doc_indices])
        """
        if metadata_keys is None:
            metadata_keys = []
        return self.batched_top_activations_and_metadatas(
            features=self.get_family_psuedofeature_tensors(
                families=families, aggregation_method=aggregation_method
            ),
            p=p,
            k=k,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_acts_sparse=return_acts_sparse,
            return_doc_indices=return_doc_indices,
            str_metadatas=str_metadatas,
            parallel=parallel,
            max_workers=max_workers,
        )

    def batched_chill_top_activations(
        self: "Evaluation",
        features: list[int | FilteredTensor],
        p: float | None = None,
        k: int | None = None,
    ) -> list:
        """
        Get TopActivations objects for multiple features efficiently.

        This is the "chill" version that returns TopActivations objects
        directly without the legacy tuple format.

        Args:
            features: List of feature IDs or FilteredTensor objects
            p: Proportion of top activations (alternative to k)
            k: Number of top activations

        Returns:
            List of TopActivations objects
        """
        if not features:
            return []

        feature_objs = Feature.make_batch(src_eval=self, features=features)
        return Feature.batched_top_activations(
            features=feature_objs,
            p=p,
            k=k,
        )

    @staticmethod
    def _pk_to_k(p: float | None, k: int | None, quantity: int) -> int:
        if (p is None) == (k is None):
            raise ValueError("Exactly one of p and k must be set")
        if p is not None and not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        if k is None:
            k = int(quantity * p)
        if k <= 0:
            raise ValueError("k must be positive")
        return min(k, quantity)

    def get_docs_and_metadatas(
        self: "Evaluation",
        doc_indices: Tensor,
        metadata_keys: list[str],
        return_str_docs: bool,
        return_str_metadatas: bool,
    ):
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]
        metadatas = {
            key: self._root_metadatas[key][doc_indices] for key in metadata_keys
        }
        if return_str_metadatas:
            metadatas = self._root_metadatas.translate(metadatas)
        return docs, metadatas
        return docs, metadatas

    def get_docs_acts_metadatas(
        self: "Evaluation",
        doc_indices: Tensor,
        features: list[FilteredTensor],
        metadata_keys: list[str],
        return_str_docs: bool,
        str_metadatas: bool,
    ):
        acts = [f.index_select(doc_indices, dim=0) for f in features]
        docs = self.docstrs[doc_indices] if return_str_docs else self.docs[doc_indices]

        docs, metadatas = self.get_docs_and_metadatas(
            doc_indices,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            return_str_metadatas=str_metadatas,
        )

        return docs, acts, metadatas

    def seq_agg_feat(
        self: "Evaluation",
        feature_id: int | None = None,
        feature: FilteredTensor | None = None,
        agg: str = "max",
        docs_filter: bool = True,
    ) -> FilteredTensor:
        if (feature_id is None) == (feature is None):
            raise ValueError("Exactly one of feat_id and feature must be set")
        if feature is None:
            assert feature_id is not None
            feature = self.features[feature_id]
        assert isinstance(feature, FilteredTensor)
        if docs_filter:
            feature = feature.filter_inactive_docs()
        if agg == "max":
            return feature.to_filtered_like_self(
                feature.value.to_dense().max(dim=1).values, ndim=1
            )
        elif agg == "sum":
            return feature.to_filtered_like_self(
                feature.value.to_dense().sum(dim=1), ndim=1
            )
        else:
            raise ValueError(f"Invalid aggregation: {agg}")

    def top_overlapped_feature_family_documents(
        self: "Evaluation",
        families: list[Family],
        p: float | None = None,
        k: int | None = None,
        metadata_keys: list[str] | None = None,
        return_str_docs: bool = False,
        str_metadatas: bool = False,
    ):
        if metadata_keys is None:
            metadata_keys = []
        if len(families) == 0:
            return [], [], [], []
        famfeats = self.get_family_psuedofeature_tensors(families=families)
        doc_acts = [self.seq_agg_feat(feature=f, agg="sum") for f in famfeats]
        agg_mask = doc_acts[0].filter.mask.clone()
        for da in doc_acts[1:]:
            agg_mask &= da.filter.mask
        filt_da = [da.mask_by_other(agg_mask, presliced=True) for da in doc_acts]
        agg_doc_score = filt_da[0].to(self.cuda).clone().to_dense()
        for da in filt_da[1:]:
            agg_doc_score *= da.to(self.cuda)
        assert agg_doc_score.ndim == 1
        if agg_doc_score.sum() == 0:
            agg_doc_score = filt_da[0].to(self.cuda).clone().to_dense()
            for da in filt_da[1:]:
                agg_doc_score += da.to(self.cuda)
        agg_doc = FilteredTensor.from_value_and_mask(
            value=agg_doc_score, mask_obj=agg_mask
        )

        k = self._pk_to_k(p, k, agg_doc_score.shape[0])
        if k == 0:
            return [], [[] for _ in range(len(families))], [], []
        topk = agg_doc.value.topk(k, sorted=True)
        top_outer_indices = agg_doc.externalize_indices(topk.indices.unsqueeze(0))
        doc_indices = top_outer_indices[0].to(self.cuda)
        docs, acts, metadatas = self.get_docs_acts_metadatas(
            doc_indices,
            features=famfeats,
            metadata_keys=metadata_keys,
            return_str_docs=return_str_docs,
            str_metadatas=str_metadatas,
        )
        return docs, acts, metadatas, doc_indices
