from typing import TYPE_CHECKING

import torch
import tqdm

from saeco.evaluation.fastapi_models.families_draft import (
    Family,
)
from saeco.evaluation.filtered import FilteredTensor

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
    ):
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
        )

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
        agg_doc = FilteredTensor.from_value_and_mask(value=agg_doc_score, mask=agg_mask)

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
