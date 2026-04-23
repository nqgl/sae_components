"""Shared prep for invoking the subject LLM.

Both the training-data activation-extraction path (`acts_data.py`) and the
reconstruction-loss evaluation path (`trainer/recons.py`, `matryoshka_clt.py`)
need to turn "whatever came out of the piler" into "things I can pass to
`model.trace(*args, **kwargs)` plus a mask of real (non-pad) positions".

Centralizing this here keeps the two paths behaviourally consistent — in
particular, it ensures the recons path honors `attention_mask_column_name`
and the HF `attention_mask` forward kwarg the same way acts generation does.
"""

from typing import TYPE_CHECKING, Any

import torch
from attrs import define

from saeco.data.dict_batch import DictBatch

if TYPE_CHECKING:
    from saeco.data.config.data_cfg import DataConfig


@define(kw_only=True)
class SubjectBatchInputs:
    """Everything needed to invoke the subject LLM on a batch (or a chunk of
    one).

    - `args`, `kwargs`: pass directly into `model.trace(*args, **kwargs)`.
    - `tokens`: the raw int64 token tensor `[batch, seq_len]` — use this for
      BOS detection, teacher-forced target construction (CE loss), or any
      other per-position reference. `None` only for exotic model backends
      whose batch shape doesn't have a single canonical "tokens" tensor.
    - `mask`: optional bool mask `[batch, seq_len]`. True at positions whose
      per-position quantities (activations, predictions) should count.
      Combines model-specific `create_acts_mask` with the dataset's
      `attention_mask_column_name`.
    """

    args: list[Any]
    kwargs: dict[str, Any]
    tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None


def transform_batch(data_cfg: "DataConfig", batch: Any) -> Any:
    """Run the model backend's `input_data_transform` on a raw piled batch.

    Call this once per full batch; then iterate chunks and call
    `prepare_subject_chunk` on each one. (Idempotent for HF, but potentially
    expensive for model-specific transforms — don't re-call per chunk.)
    """
    return data_cfg.model_cfg.model_load_cfg.input_data_transform(batch)


def prepare_subject_chunk(
    data_cfg: "DataConfig",
    tx_chunk: torch.Tensor | DictBatch,
    seq_len: int | None = None,
) -> SubjectBatchInputs:
    """Build a `SubjectBatchInputs` from an already-transformed chunk.

    Vanilla `DictBatch` (what the on-the-fly PAD tokenizer writes) is
    handled here: the tokens tensor is extracted via
    `data_cfg.tokens_column_name`, and if `data_cfg.attention_mask_column_name`
    is set, its mask is forwarded under HF's fixed `"attention_mask"` forward
    kwarg name. Model-specific `DictBatch` subclasses (ComLM's
    `XRNoisedBatch`, etc.) are routed through their own `unpack_model_inputs`
    unchanged.
    """
    model_load_cfg = data_cfg.model_cfg.model_load_cfg
    extra_kwargs = dict(data_cfg.model_cfg.model_kwargs)

    tokens_tensor: torch.Tensor | None = None
    attention_mask_bool: torch.Tensor | None = None
    inner_input = tx_chunk

    if type(tx_chunk) is DictBatch:
        if data_cfg.attention_mask_column_name is not None:
            am = tx_chunk.get(data_cfg.attention_mask_column_name)
            if am is not None:
                extra_kwargs["attention_mask"] = am.to(dtype=torch.long)
                attention_mask_bool = am.bool()
        tokens_tensor = tx_chunk[data_cfg.tokens_column_name]
        inner_input = tokens_tensor
    elif isinstance(tx_chunk, torch.Tensor):
        tokens_tensor = tx_chunk
    else:
        raise TypeError(f"Unknown batch type: {type(tx_chunk)}")

    assert inner_input is not None

    args, kwargs = model_load_cfg.unpack_model_inputs(inner_input, extra_kwargs)

    effective_seq_len = seq_len if seq_len is not None else data_cfg.seq_len
    model_mask = None
    if effective_seq_len is not None:
        model_mask = model_load_cfg.create_acts_mask(tx_chunk, effective_seq_len)

    mask: torch.Tensor | None
    if model_mask is not None and attention_mask_bool is not None:
        am_sliced = attention_mask_bool
        if effective_seq_len is not None:
            am_sliced = am_sliced[:, :effective_seq_len]
        mask = model_mask & am_sliced
    elif model_mask is not None:
        mask = model_mask
    elif attention_mask_bool is not None:
        mask = attention_mask_bool
        if effective_seq_len is not None:
            mask = mask[:, :effective_seq_len]
    else:
        mask = None

    return SubjectBatchInputs(args=args, kwargs=kwargs, tokens=tokens_tensor, mask=mask)


def prepare_subject_batch(
    data_cfg: "DataConfig",
    batch: torch.Tensor | DictBatch,
    seq_len: int | None = None,
) -> SubjectBatchInputs:
    """One-shot transform + prepare. For callers that don't stream sub-chunks
    (e.g. the recons evaluator, which runs one batch at a time)."""
    tx = data_cfg.model_cfg.model_load_cfg.input_data_transform(batch)
    return prepare_subject_chunk(data_cfg, tx, seq_len=seq_len)
