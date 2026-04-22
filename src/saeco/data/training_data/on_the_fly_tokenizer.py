from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import tqdm
from attrs import define

from saeco.data.config.split_config import SplitConfig
from saeco.data.config.tokenization_config import PackingMode, TokenizationMode
from saeco.data.dict_batch import DictBatch

if TYPE_CHECKING:
    from saeco.data.config.data_cfg import DataConfig


def _strip_historical_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gemma-4: historical assistant thoughts must not be re-fed in multi-turn.

    Drops any `thinking` field and any content item whose `type` is `thinking`
    from every assistant turn except the last. The final assistant turn is left
    untouched so that if we're using it as training data the reasoning trace is
    preserved for tokenization.
    """
    last_assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") in ("assistant", "model"):
            last_assistant_idx = i

    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") not in ("assistant", "model") or i == last_assistant_idx:
            out.append(m)
            continue
        cleaned = {k: v for k, v in m.items() if k != "thinking"}
        content = cleaned.get("content")
        if isinstance(content, list):
            cleaned["content"] = [
                c for c in content if not (isinstance(c, dict) and c.get("type") == "thinking")
            ]
        out.append(cleaned)
    return out


@define
class OnTheFlyTokenizer:
    cfg: "DataConfig"
    split: SplitConfig
    tokenizer: Any

    def _iter_doc_ids(self) -> Iterator[list[int]]:
        tcfg = self.cfg.tokenization
        dataset = self.cfg.load_dataset_from_split(self.split, to_torch=False)

        if tcfg.mode == TokenizationMode.RAW_TEXT:
            text_col = tcfg.text_column
            tok_kwargs = {
                "add_special_tokens": True,
                "return_attention_mask": False,
                "truncation": False,
                **tcfg.tokenizer_kwargs,
            }
            tokenizer = self.tokenizer

            def tokenize_fn(batch):
                out = tokenizer(batch[text_col], **tok_kwargs)
                return {"input_ids": out["input_ids"]}

            mapped = dataset.map(
                tokenize_fn,
                batched=True,
                batch_size=tcfg.map_batch_size,
                num_proc=tcfg.map_num_proc,
                remove_columns=list(dataset.column_names),
                desc="tokenizing (raw_text)",
            )
            for row in mapped:
                ids = row["input_ids"]
                if len(ids) >= tcfg.min_seq_len:
                    yield list(ids)
            return

        if tcfg.mode == TokenizationMode.CONVERSATION:
            messages_col = tcfg.messages_column
            chat_kwargs = {
                "tokenize": True,
                "return_dict": False,
                **tcfg.chat_template_kwargs,
            }
            for row in tqdm.tqdm(dataset, desc="tokenizing (conversation)"):
                messages = row[messages_col]
                if tcfg.strip_historical_thinking:
                    messages = _strip_historical_thinking(messages)
                ids = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) >= tcfg.min_seq_len:
                    yield list(ids)
            return

        raise ValueError(f"OnTheFlyTokenizer cannot handle mode {tcfg.mode}")

    def iter_tensor_batches(self, yield_rows_per_batch: int) -> Iterator[torch.Tensor]:
        """PACK / TRUNCATE paths. Yields int64 [N<=rows_per_batch, seq_len] tensors."""
        tcfg = self.cfg.tokenization
        assert tcfg.packing in (PackingMode.PACK, PackingMode.TRUNCATE)
        seq_len = self.cfg.seq_len
        assert seq_len is not None, "seq_len must be set for on-the-fly tokenization"

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "pad_token_id", None)
        if eos_id is None:
            eos_id = 0

        buffer: list[int] = []
        rows: list[list[int]] = []

        def flush_batch() -> torch.Tensor:
            t = torch.tensor(rows, dtype=torch.int64)
            rows.clear()
            return t

        for ids in self._iter_doc_ids():
            if tcfg.packing == PackingMode.TRUNCATE:
                if len(ids) >= seq_len:
                    rows.append(ids[:seq_len])
                else:
                    continue
            else:  # PACK
                buffer.extend(ids)
                buffer.append(eos_id)
                while len(buffer) >= seq_len:
                    rows.append(buffer[:seq_len])
                    del buffer[:seq_len]
            if len(rows) >= yield_rows_per_batch:
                yield flush_batch()

        if rows:
            yield flush_batch()

    def iter_dict_batches(self, yield_rows_per_batch: int) -> Iterator[DictBatch]:
        """PAD path. Yields DictBatch(input_ids=int64, attention_mask=bool) rows."""
        tcfg = self.cfg.tokenization
        assert tcfg.packing == PackingMode.PAD
        seq_len = self.cfg.seq_len
        assert seq_len is not None

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)

        input_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []

        def flush_batch() -> DictBatch:
            ids_t = torch.tensor(input_rows, dtype=torch.int64)
            mask_t = torch.tensor(mask_rows, dtype=torch.bool)
            input_rows.clear()
            mask_rows.clear()
            return DictBatch(data={"input_ids": ids_t, "attention_mask": mask_t})

        for ids in self._iter_doc_ids():
            ids = list(ids[:seq_len])
            length = len(ids)
            if length < seq_len:
                ids = ids + [pad_id] * (seq_len - length)
            mask = [True] * length + [False] * (seq_len - length)
            input_rows.append(ids)
            mask_rows.append(mask)
            if len(input_rows) >= yield_rows_per_batch:
                yield flush_batch()

        if input_rows:
            yield flush_batch()

    def estimate_num_tokens(self) -> int:
        """Best-effort estimate used to size the piler up-front.

        Prefers `split.tokens_from_split` when the user set it. Otherwise falls
        back to a heuristic from dataset row count. Overshooting is harmless
        (piler accepts empty piles); undershoot caps the total tokens piled.
        """
        if self.split.tokens_from_split is not None:
            return self.split.tokens_from_split
        dataset = self.cfg.load_dataset_from_split(self.split, to_torch=False)
        nrows = len(dataset)
        tcfg = self.cfg.tokenization
        avg_tokens_per_row = 4096 if tcfg.mode == TokenizationMode.CONVERSATION else 1024
        return nrows * avg_tokens_per_row
