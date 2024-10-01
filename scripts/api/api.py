from fastapi import FastAPI

from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.return_types import (
    MetadataEnrichmentResults,
    MetadataLabelEnrichment,
    TopActivatingExamplesQuery,
    TopActivatingExamplesResult,
)


def create_app(root: Evaluation):
    app = FastAPI()

    @app.put("/top_activating_examples")
    def get_top_activating_examples(
        query: TopActivatingExamplesQuery,
    ) -> TopActivatingExamplesResult:
        docs, acts, metadatas, doc_indices = root.top_activations_and_metadatas(
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

    return app
