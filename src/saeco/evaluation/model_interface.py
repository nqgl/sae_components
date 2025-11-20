from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from saeco.data.dict_batch import DictBatch


class ModelEvalAdapter[BatchT, OutputT](ABC):
    """
    Small shim that lets Evaluation remain agnostic to the underlying model/batch
    structure (language models vs. comlm gene models, etc.).
    """

    def __init__(self, model_kwargs: dict[str, Any] | None = None):
        self.model_kwargs = model_kwargs or {}

    def make_batch(
        self,
        tokens: torch.Tensor | DictBatch | BatchT,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> BatchT:
        """
        Convert stored tokens + optional metadata into the model's expected batch type.
        Default: pass through.
        """
        return tokens  # type: ignore[return-value]

    def trace(self, model, batch: BatchT):
        """
        Wraps the model trace call so callers don't need to know the signature.
        """
        return model.trace(batch, **self.model_kwargs)

    def unwrap_output(self, output: OutputT) -> OutputT:
        """
        Hook point in case an adapter needs to post-process the saved output node.
        """
        return output

    @abstractmethod
    def get_logits(self, output: OutputT) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss(self, model, output: OutputT, batch: BatchT) -> torch.Tensor: ...


class LanguageModelEvalAdapter(ModelEvalAdapter[torch.Tensor, Any]):
    """
    Adapter for standard language models that take token tensors.
    """

    def get_logits(self, output: Any) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits  # type: ignore[return-value]
        if torch.is_tensor(output):
            return output
        raise TypeError(f"Cannot extract logits from output type {type(output)}")

    def compute_loss(self, model, output: Any, batch: torch.Tensor) -> torch.Tensor:
        logits = self.get_logits(output)
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits shape (batch, seq, vocab); got {logits.shape}"
            )
        # Shift for next-token prediction
        l = logits[:, :-1]
        tgt = batch[:, 1:]
        return torch.nn.functional.cross_entropy(
            l.reshape(-1, l.shape[-1]),
            tgt.reshape(-1),
        )


class ComlmEvalAdapter(ModelEvalAdapter[Any, Any]):
    """
    Adapter for comlm/XR models. Uses NoisedBatch inputs and the model's native loss.
    """

    def __init__(self, model_kwargs: dict[str, Any] | None = None):
        super().__init__(model_kwargs=model_kwargs)

    def make_batch(
        self,
        tokens: torch.Tensor | DictBatch | Any,
        metadata: dict[str, torch.Tensor] | None = None,
    ):
        try:
            from comlm.datasource.training_batch import NoisedBatch
        except Exception as exc:  # pragma: no cover - only catches missing comlm
            raise ImportError(
                "comlm must be installed to use ComlmEvalAdapter"
            ) from exc

        metadata = metadata or {}
        if isinstance(tokens, NoisedBatch):
            return tokens

        counts = metadata.get("counts")
        loss_mask = metadata.get("loss_mask")
        attention_mask = metadata.get("attention_mask")
        ranks = metadata.get("ranks")
        md = metadata.get("metadata")

        if isinstance(tokens, DictBatch):
            clean_tokens = tokens.get("clean_tokens", None)
            if clean_tokens is None:
                # fall back to the first tensor entry
                clean_tokens = next(iter(tokens.values()))
            counts = counts if counts is not None else tokens.get("counts", None)
            loss_mask = (
                loss_mask if loss_mask is not None else tokens.get("loss_mask", None)
            )
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else tokens.get("attention_mask", None)
            )
            md = md if md is not None else tokens.get("metadata", None)
            ranks = ranks if ranks is not None else tokens.get("ranks", None)
        else:
            clean_tokens = tokens

        assert isinstance(clean_tokens, torch.Tensor)
        if counts is None:
            counts = torch.zeros_like(clean_tokens)
        if loss_mask is None:
            loss_mask = torch.ones_like(clean_tokens, dtype=torch.bool)
        if attention_mask is None:
            attention_mask = torch.ones_like(clean_tokens, dtype=torch.bool)
        if md is None:
            md = torch.zeros_like(clean_tokens[:, :0], dtype=torch.long)

        return NoisedBatch.new_unnoised(
            clean_tokens,
            counts=counts,
            loss_mask=loss_mask,
            metadata=md,
            ranks=ranks,
            attention_mask=attention_mask,
        )

    def trace(self, model, batch: Any):
        return model.trace(batch, **self.model_kwargs)

    def get_logits(self, output: Any) -> torch.Tensor:
        if hasattr(output, "get_logits"):
            logits = output.get_logits()
            if logits is not None:
                return logits
        if hasattr(output, "logits"):
            return output.logits  # type: ignore[return-value]
        raise TypeError(f"Cannot extract logits from output type {type(output)}")

    def compute_loss(self, model, output: Any, batch: Any) -> torch.Tensor:
        if hasattr(model, "loss"):
            loss_dict = model.loss(output, batch)
            if isinstance(loss_dict, dict):
                if "total" in loss_dict:
                    return loss_dict["total"]
                # pick the first tensor if "total" isn't present
                for v in loss_dict.values():
                    if torch.is_tensor(v):
                        return v
            if torch.is_tensor(loss_dict):
                return loss_dict
        raise AttributeError(
            "Model does not expose a compatible loss(output, batch) method"
        )
