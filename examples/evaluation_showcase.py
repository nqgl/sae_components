from __future__ import annotations

import os
from pathlib import Path

import torch

from saeco.evaluation.evaluation import Evaluation
from saeco.data.dict_batch import DictBatch


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def load_eval() -> Evaluation:
    """
    Load an Evaluation from an on-disk cache.
    Set SAECO_CACHE to either:
      - a cache dirname (resolved under ~/workspace/cached_sae_acts/<name>)
      - an absolute/relative path to the cache directory
    """
    cache = os.environ.get("SAECO_CACHE", "test")
    return Evaluation.from_cache_name(cache)


def ensure_comlm_metadata(eval: Evaluation) -> list[str]:
    """
    If this run looks like a comlm-style run with a metadata tokenizer and a DictBatch field
    called "metadata", persist metadata tensors into eval.metadatas.

    Returns: metadata keys list (possibly empty if not applicable).
    """
    # Detect metadata_tokenizer (comlm arch usually has this)
    metadata_tokenizer = getattr(getattr(eval.architecture, "data", None), "metadata_tokenizer", None)
    if metadata_tokenizer is None:
        return []

    keys = list(getattr(metadata_tokenizer, "tokenizers", {}).keys())
    if not keys:
        return []

    # If already present, do nothing.
    missing = [k for k in keys if k not in eval.metadatas]
    if not missing:
        return keys

    _print_header("Initializing metadata tensors (one-time)")

    # Build a single tensor shaped (num_docs, num_keys) by streaming chunks.
    builder = eval.metadata_builder(dtype=torch.long, device="cpu", item_size=(len(keys),))
    for chunk in builder:
        tokens = chunk.tokens.value
        if not isinstance(tokens, DictBatch):
            raise TypeError("Expected DictBatch tokens for comlm metadata init")
        if "metadata" not in tokens:
            raise KeyError('DictBatch is missing "metadata" field')
        builder << tokens["metadata"]

    full = builder.value  # (num_docs, num_keys)

    # Store each column under its own name.
    for i, k in enumerate(keys):
        tok = metadata_tokenizer.tokenizers[k]
        eval.metadatas[k] = full[:, i]
        # Include PAD/UNK then tokenizer vocab.
        eval.metadatas.set_str_translator(k, {"<<PAD>>": 0, "<<UNK>>": 1, **tok.tokens})

    print(f"Stored metadatas: {keys}")
    return keys


def show_basic(eval: Evaluation) -> None:
    _print_header("Basic info")
    print("cache path:", eval.path)
    print("num_docs:", eval.num_docs)
    print("seq_len:", eval.seq_len)
    print("d_dict:", eval.d_dict)
    print("device:", eval.cuda)

    _print_header("First docs (detokenized)")
    # docs may be Tensor or DictBatch; detokenize handles both.
    doc0 = eval.docs[0]
    doc1 = eval.docs[1]
    print("doc[0]:", eval.detokenize(doc0))
    print("doc[1]:", eval.detokenize(doc1))


def showcase_feature(eval: Evaluation, feature_id: int, metadata_keys: list[str]) -> None:
    _print_header(f"Feature {feature_id}: top activations")

    feat = eval.get_feature(feature_id)
    top = feat.top_activations(k=8)

    print("Top doc indices:", top.doc_selection.doc_indices.tolist())
    print("Example doc strings:")
    doc_strs = top.doc_strs
    # doc_strs might be list[list[str]] or list[str]; print a compact view
    for i, s in enumerate(doc_strs[:3] if isinstance(doc_strs, list) else [doc_strs]):
        print(f"- #{i}:", s if isinstance(s, str) else "".join(s))

    if metadata_keys:
        _print_header("Metadata enrichments")
        enrich = eval.top_activations_metadata_enrichments(
            feature=feature_id,
            metadata_keys=metadata_keys,
            p=0.10,
            str_label=True,
        )
        # This is a pydantic model; printing keeps it readable enough.
        print(enrich)


def showcase_cosims(eval: Evaluation) -> None:
    _print_header("Activation cosine similarities")
    cos = eval.activation_cosims(out_device="cpu", blocks_per_dim=2)
    print("cosims shape:", tuple(cos.shape))
    # quick sanity
    diag = cos.diag()
    print("diag mean (ignoring NaN):", diag[~diag.isnan()].mean().item())


def main() -> None:
    eval = load_eval()

    show_basic(eval)

    # Optional: auto-initialize metadata if this looks like a comlm run.
    metadata_keys = ensure_comlm_metadata(eval)

    # Pick a “reasonable” feature to show.
    showcase_feature(eval, feature_id=7, metadata_keys=metadata_keys)

    showcase_cosims(eval)

    # OPTIONAL: patching demo (can be slow / requires model interface)
    if os.environ.get("SAECO_PATCHING_DEMO", "0") == "1":
        _print_header("Patching demo: ablate feature at its max-activation position")
        # This is intentionally minimal; for deeper attribution, use the patching helpers.
        feature_id = 7
        feat = eval.features[feature_id].to(eval.cuda).filter_inactive_docs()
        idx = feat.indices()
        # pick one occurrence (doc, seq, feat) from sparse indices
        doc_id = int(idx[0, 0].item())
        seq_pos = int(idx[1, 0].item())
        tokens = eval.docs[doc_id : doc_id + 1]
        tokens = tokens.to(eval.cuda) if torch.is_tensor(tokens) else tokens.to(eval.cuda)

        def patch_fn(acts: torch.Tensor) -> torch.Tensor:
            acts = acts.clone()
            acts[0, seq_pos, feature_id] = 0
            return acts

        diff = eval.patchdiff(tokens, patch_fn, invert=True, doc_indices=torch.tensor([doc_id], device=eval.cuda))
        print("patchdiff:", diff.shape)


if __name__ == "__main__":
    main()
