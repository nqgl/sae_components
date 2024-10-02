from fastapi import FastAPI

from .evaluation import Evaluation
from .fastapi_models import (
    FeatureActiveDocsRequest,
    FeatureActiveDocsResponse,
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


def create_app(root: Evaluation):
    app = FastAPI()

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
        )
        if not query.return_str_docs:
            docs = docs.tolist()
        acts = acts.to_dense()
        acts = acts.tolist()
        metadatas = [m.tolist() for m in metadatas]
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
        return TokenEnrichmentResponse(
            results=[
                TokenEnrichmentResponseItem(
                    tokstr=tokstr,
                    token=token,
                    count=count,
                    normalized_count=normalized_count,
                    score=score,
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
            num_active_docs=ev.features[query.feature]
            .filter_inactive_docs()
            .filter.mask.sum()
            .item()
        )

    return app
