from __future__ import annotations

from typing import TYPE_CHECKING, Any

from attrs import define
from torch import Tensor

if TYPE_CHECKING:
    from .evaluation import Evaluation


@define(slots=True)
class DecodedTextView:
    """
    Indexable view returning decoded text for tokens.

    Examples:
      eval.text[0]        -> str
      eval.text[0:10]     -> list[str]
      eval.text[doc_ids]  -> list[str]
    """

    eval: Evaluation
    skip_special_tokens: bool = False

    def __getitem__(self, idx: Any) -> str | list[str]:
        tokens = self.eval.tokens[idx]
        return self.eval.decode_text(
            tokens, skip_special_tokens=self.skip_special_tokens
        )


@define(slots=True)
class TokenStringsView:
    """
    Indexable view returning token strings (not joined text).

    Examples:
      eval.token_strs[0]        -> list[str]
      eval.token_strs[0:3]      -> list[list[str]]
      eval.token_strs[doc_ids]  -> list[list[str]]
    """

    eval: Evaluation

    def __getitem__(self, idx: Any) -> str | list[str] | list[list[str]]:
        tokens = self.eval.samples[idx]
        return self.eval.detokenize(tokens)


@define(slots=True)
class MetadataView:
    """
    Evaluation-scoped metadata accessor.

    Works on filtered evals too: it returns metadata aligned to the *current*
    evaluation's doc space.

    Examples:
      eval.metadata["source"][:10]
      eval.metadata.as_str("language", eval.metadata["language"][:10])
    """

    eval: Evaluation

    def __getitem__(self, key: str) -> Tensor:
        return self.eval.metadata_store[key]

    def as_str(self, key: str, values: Tensor | None = None) -> list[str]:
        vals = self[key] if values is None else values
        # Translator lives on root metadatas.
        md = self.eval._root_metadatas.get(key)
        return md.strlist(vals.cpu())
