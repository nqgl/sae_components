import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import Field

from saeco.sweeps import SweepableConfig


class TokenizationMode(str, Enum):
    PRETOKENIZED = "pretokenized"
    RAW_TEXT = "raw_text"
    CONVERSATION = "conversation"


class PackingMode(str, Enum):
    PACK = "pack"
    PAD = "pad"
    TRUNCATE = "truncate"


class TokenizationConfig(SweepableConfig):
    mode: TokenizationMode = TokenizationMode.PRETOKENIZED
    text_column: str = "text"
    messages_column: str = "messages"
    packing: PackingMode = PackingMode.PACK

    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)

    strip_historical_thinking: bool = True

    map_batch_size: int = 1000
    map_num_proc: int | None = 4
    min_doc_tokens: int = 8

    template_version_tag: str | None = None

    def idstr_fragment(self) -> str:
        if self.mode == TokenizationMode.PRETOKENIZED:
            return ""
        parts = [self.mode.value, self.packing.value]
        if self.chat_template_kwargs:
            parts.append(
                "ck_"
                + hashlib.sha1(
                    json.dumps(self.chat_template_kwargs, sort_keys=True).encode()
                ).hexdigest()[:10]
            )
        if self.tokenizer_kwargs:
            parts.append(
                "tk_"
                + hashlib.sha1(
                    json.dumps(self.tokenizer_kwargs, sort_keys=True).encode()
                ).hexdigest()[:10]
            )
        if self.template_version_tag:
            parts.append(f"tv_{self.template_version_tag}")
        return "_" + "_".join(parts)
