from fastapi import FastAPI

from saeco.evaluation.evaluation import Evaluation
from .fastapi_models import (
    MetadataEnrichmentRequest,
    MetadataEnrichmentResponse,
    TopActivatingExamplesQuery,
    TopActivatingExamplesResult,
)


def create_app(root: Evaluation):
    app = FastAPI()

    def get_eval(filter_id):
        if filter_id is None:
            print("filter_id is None")
            return root
        if filter_id == "root":
            return root
        return root.open_filtered(filter_id)

    @app.put("/{filter_id}/top_activating_examples")
    def get_top_activating_examples(
        filter_id: str | None,
        query: TopActivatingExamplesQuery,
    ) -> TopActivatingExamplesResult:
        evaluation = get_eval(filter_id=filter_id)
        docs, acts, metadatas, doc_indices = evaluation.top_activations_and_metadatas(
            query.feature,
            p=query.p,
            k=query.k,
            metadata_keys=query.metadata_keys,
            return_str_docs=query.return_str_docs,
            # return_acts_sparse=query.return_acts_sparse,
        )
        if not query.return_str_docs:
            docs = docs.tolist()
        acts = acts.to_dense()
        acts = acts.tolist()
        return TopActivatingExamplesResult(
            docs=docs,
            acts=acts,
            metadatas=[m.tolist() for m in metadatas],
            doc_indices=doc_indices.tolist(),
        )

    @app.put("/{filter_id}/metadata_enrichment")
    def get_metadata_enrichment(
        filter_id: str | None,
        query: MetadataEnrichmentRequest,
    ) -> MetadataEnrichmentResponse:
        evaluation = get_eval(filter_id=filter_id)
        return evaluation.top_activations_metadata_enrichments(
            feature=query.feature,
            metadata_keys=query.metadata_keys,
            p=query.p,
            k=query.k,
            str_label=query.str_label,
            normalize_metadata_frequencies=query.normalize_metadata_frequencies,
        )

    return app
