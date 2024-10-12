import json

from pathlib import Path

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from .evaluation import Evaluation
from .fastapi_models import (
    CoActivatingFeature,
    CoActivationRequest,
    CoActivationResponse,
    FeatureActiveDocsRequest,
    FeatureActiveDocsResponse,
    GeneInfo,
    MetadataEnrichmentRequest,
    MetadataEnrichmentResponse,
    TokenEnrichmentMode,
    TokenEnrichmentRequest,
    TokenEnrichmentResponse,
    TokenEnrichmentResponseItem,
    TopActivatingExamplesQuery,
    TopActivatingExamplesResult,
    TopActivationResultEntry,
)

from .fastapi_models.families_draft import (
    Family,
    FamilyLevel,
    FamilyTopActivatingExamplesQuery,
    Feature,
    GetFamiliesRequest,
    GetFamiliesResponse,
    TopFamilyOverlappingExamplesResponseDoc,
)
from .fastapi_models.Feature import Feature

gene_conversions_path = (
    Path.home() / "workspace" / "cached_sae_acts" / "class_conversion.json"
)

gene_conversions = {
    k: GeneInfo.model_validate(v)
    for k, v in json.loads(gene_conversions_path.read_text()).items()
}


def create_app(root: Evaluation):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    @app.put("/top_activating_examples")
    def get_top_activating_examples(
        query: TopActivatingExamplesQuery,
    ) -> TopActivatingExamplesResult:
        evaluation = query.filter(root)
        docs, acts, metadatas, doc_indices = evaluation.top_activations_and_metadatas(
            query.feature,
            p=query.p,
            k=query.k,
            metadata_keys=query.metadata_keys,
            return_str_docs=query.return_str_docs,
            str_metadatas=query.return_str_metadatas,
        )
        if not query.return_str_docs:
            docs = docs.tolist()
        acts = acts.to_dense()
        acts = acts.tolist()
        metadatas = [metadatas[k] for k in query.metadata_keys]
        metadatas = [m if isinstance(m, list) else m.tolist() for m in metadatas]
        if len(metadatas) == 0:
            metadatas = [[] for _ in range(len(docs))]
        else:
            metadatas = [
                [metadatas[i][j] for i in range(len(metadatas))]
                for j in range(len(metadatas[0]))
            ]
        assert len(docs) == len(acts) == len(metadatas) == len(doc_indices), (
            len(docs),
            len(acts),
            len(metadatas),
            len(doc_indices),
        )

        return TopActivatingExamplesResult(
            entries=[
                TopActivationResultEntry(
                    doc=doc,
                    metadatas=md,
                    acts=act,
                    doc_index=doc_id,
                )
                for doc, act, md, doc_id in zip(
                    docs, acts, metadatas, doc_indices.tolist()
                )
            ]
        )
        # return TopActivatingExamplesResult(
        #     docs=docs,
        #     acts=acts,
        #     metadatas=[m.tolist() for m in metadatas],
        #     doc_indices=doc_indices.tolist(),
        # )

    @app.put("/metadata_enrichment")
    def get_metadata_enrichment(
        query: MetadataEnrichmentRequest,
    ) -> MetadataEnrichmentResponse:
        ev = query.filter(root)
        return ev.top_activations_metadata_enrichments(
            feature=query.feature,
            metadata_keys=query.metadata_keys,
            p=query.p,
            k=query.k,
            str_label=query.str_label,
        )

    @app.put("/token_enrichment")
    def get_token_enrichment(query: TokenEnrichmentRequest) -> TokenEnrichmentResponse:
        ev = query.filter(root)
        tokens, counts, normalized_counts, scores = (
            ev.top_activations_token_enrichments(
                feature=query.feature,
                mode=query.mode,
                p=query.p,
                k=query.k,
                sort_by=query.sort_by,
            )
        )
        tokstrs = ev.detokenize(tokens)
        assert (
            len(tokens)
            == len(counts)
            == len(normalized_counts)
            == len(scores)
            == len(tokstrs)
        ), (len(tokens), len(counts), len(normalized_counts), len(scores), tokstrs)
        if query.num_top_tokens:
            tokens = tokens[: query.num_top_tokens]
            counts = counts[: query.num_top_tokens]
            normalized_counts = normalized_counts[: query.num_top_tokens]
            scores = scores[: query.num_top_tokens]
            tokstrs = tokstrs[: query.num_top_tokens]
        return TokenEnrichmentResponse[GeneInfo](
            results=[
                TokenEnrichmentResponseItem[GeneInfo](
                    tokstr=tokstr,
                    token=token,
                    count=count,
                    normalized_count=normalized_count,
                    score=score,
                    info=gene_conversions[tokstr],
                )
                for tokstr, token, count, normalized_count, score in zip(
                    tokstrs,
                    tokens.tolist(),
                    counts.tolist(),
                    normalized_counts.tolist(),
                    scores.tolist(),
                )
            ]
        )

    @app.put("/feature_active_docs_count")
    def get_feature_active_docs_count(
        query: FeatureActiveDocsRequest,
    ) -> FeatureActiveDocsResponse:
        ev = query.filter(root)
        return FeatureActiveDocsResponse(
            num_active_docs=ev.num_active_docs_for_feature(query.feature)
        )

    @app.put("/top_coactivating_features")
    def get_top_coactive_features(query: CoActivationRequest):
        ev = query.filter(root)
        ids, values = ev.top_coactivating_features(
            feature_id=query.feature_id,
            top_n=query.n,
            mode=query.mode,
        )
        l = []
        for i, v in zip(ids, values):
            l.append(
                CoActivatingFeature(feature_id=i.item(), coactivation_level=v.item())
            )
        return CoActivationResponse(results=l)

    @app.put("/get_families")
    def get_families(query: GetFamiliesRequest) -> GetFamiliesResponse:
        ev = query.filter(root)
        return ev.cached_call.get_feature_families()

    @app.put("/family_top_activating_examples")
    def get_family_top_activating_examples(
        query: FamilyTopActivatingExamplesQuery,
    ) -> list[TopActivatingExamplesResult]:
        ev = query.filter(root)
        all_families = ev.cached_call.get_feature_families()

        families = [
            all_families.levels[family.level].families[family.family_id]
            for family in query.families
        ]

        batches = ev.batched_top_activations_and_metadatas_for_family(
            families=families,
            p=query.p,
            k=query.k,
            metadata_keys=query.metadata_keys,
            return_str_docs=query.return_str_docs,
            str_metadatas=query.return_str_metadatas,
        )
        out = []
        for batch in batches:
            docs, acts, metadatas, doc_indices = batch
            if not query.return_str_docs:
                docs = docs.tolist()
            acts = acts.to_dense()
            acts = acts.tolist()
            metadatas = [metadatas[k] for k in query.metadata_keys]
            metadatas = [m if isinstance(m, list) else m.tolist() for m in metadatas]
            if len(metadatas) == 0:
                metadatas = [[] for _ in range(len(docs))]
            else:
                metadatas = [
                    [metadatas[i][j] for i in range(len(metadatas))]
                    for j in range(len(metadatas[0]))
                ]
            assert len(docs) == len(acts) == len(metadatas) == len(doc_indices), (
                len(docs),
                len(acts),
                len(metadatas),
                len(doc_indices),
            )

            out.append(
                TopActivatingExamplesResult(
                    entries=[
                        TopActivationResultEntry(
                            doc=doc,
                            metadatas=md,
                            acts=act,
                            doc_index=doc_id,
                        )
                        for doc, act, md, doc_id in zip(
                            docs, acts, metadatas, doc_indices.tolist()
                        )
                    ]
                )
            )
        return out

    @app.put("/family_top_overlapping_examples")
    def get_family_top_activating_examples(
        query: FamilyTopActivatingExamplesQuery,
    ) -> list[TopFamilyOverlappingExamplesResponseDoc]:
        ev = query.filter(root)
        all_families = ev.cached_call.get_feature_families()

        families = [
            all_families.levels[family.level].families[family.family_id]
            for family in query.families
        ]

        docs, fam_acts, metadatas, doc_indices = (
            ev.top_overlapped_feature_family_documents(
                families=families,
                p=query.p,
                k=query.k,
                metadata_keys=query.metadata_keys,
                return_str_docs=query.return_str_docs,
                str_metadatas=query.return_str_metadatas,
            )
        )
        metadatas = [metadatas[k] for k in query.metadata_keys]
        metadatas = [m if isinstance(m, list) else m.tolist() for m in metadatas]
        if len(metadatas) == 0:
            metadatas = [[] for _ in range(len(docs))]
        else:
            metadatas = [
                [metadatas[i][j] for i in range(len(metadatas))]
                for j in range(len(metadatas[0]))
            ]
        fam_acts = [a.to_dense() for a in fam_acts]
        fam_acts = [a.tolist() for a in fam_acts]
        if not query.return_str_docs:
            docs = docs.tolist()

        return [
            TopFamilyOverlappingExamplesResponseDoc(
                doc=doc,
                metadatas=md,
                acts=[acts[i] for acts in fam_acts],
                doc_index=doc_id,
            )
            for i, (doc, md, doc_id) in enumerate(
                zip(docs, metadatas, doc_indices.tolist())
            )
        ]

    return app
