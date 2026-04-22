"""End-to-end smoke tests for OnTheFlyTokenizer with an in-memory HF dataset
and the (non-gated) gpt2 tokenizer. Exercises raw_text PACK and conversation
PAD paths without touching the full DataConfig cache or piler layers.
"""

from types import SimpleNamespace

import datasets
import pytest
import torch

from saeco.data.config.split_config import SplitConfig
from saeco.data.config.tokenization_config import (
    PackingMode,
    TokenizationConfig,
    TokenizationMode,
)
from saeco.data.training_data.on_the_fly_tokenizer import (
    OnTheFlyTokenizer,
    _strip_historical_thinking,
)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def _make_cfg(raw_dataset, tokenization: TokenizationConfig, seq_len: int = 32):
    """Minimal stand-in for DataConfig — OnTheFlyTokenizer only accesses
    load_dataset_from_split(), tokenization, and seq_len."""
    cfg = SimpleNamespace()
    cfg.tokenization = tokenization
    cfg.seq_len = seq_len
    cfg.load_dataset_from_split = lambda split, to_torch=True: raw_dataset
    return cfg


def test_raw_text_pack(gpt2_tokenizer, tmp_path):
    sentences = [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Hello, world! This is a tokenization test. " * 4,
        "Sparse autoencoders learn features. " * 4,
    ]
    hf = datasets.Dataset.from_dict({"text": sentences})
    # Single proc to avoid subprocess pickling problems with SimpleNamespace.
    tcfg = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT,
        packing=PackingMode.PACK,
        map_num_proc=None,
        map_batch_size=2,
    )
    otf = OnTheFlyTokenizer(
        cfg=_make_cfg(hf, tcfg, seq_len=16),
        split=SplitConfig(),
        tokenizer=gpt2_tokenizer,
    )

    batches = list(otf.iter_tensor_batches(yield_rows_per_batch=4))
    assert batches, "expected at least one packed batch"
    for t in batches:
        assert t.dtype == torch.int64
        assert t.shape[1] == 16
        assert t.shape[0] > 0


def test_raw_text_truncate_drops_short(gpt2_tokenizer):
    hf = datasets.Dataset.from_dict({"text": ["short", "also short"]})
    tcfg = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT,
        packing=PackingMode.TRUNCATE,
        map_num_proc=None,
        min_seq_len=0,
    )
    otf = OnTheFlyTokenizer(
        cfg=_make_cfg(hf, tcfg, seq_len=128),
        split=SplitConfig(),
        tokenizer=gpt2_tokenizer,
    )
    batches = list(otf.iter_tensor_batches(yield_rows_per_batch=4))
    # Short documents — TRUNCATE requires len >= seq_len, so no rows should be yielded.
    assert batches == []


def test_conversation_pad(gpt2_tokenizer):
    # gpt2 has no built-in chat template. Install a tiny one so the test runs
    # locally without a gated model.
    tokenizer = gpt2_tokenizer
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "<|{{m['role']}}|>{{m['content']}}<|end|>"
        "{% endfor %}"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    convos = [
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there!"},
        ],
        [
            {"role": "user", "content": "what's 2+2?"},
            {"role": "assistant", "content": "four"},
        ],
    ]
    hf = datasets.Dataset.from_dict({"messages": convos})
    tcfg = TokenizationConfig(
        mode=TokenizationMode.CONVERSATION,
        packing=PackingMode.PAD,
        messages_column="messages",
    )
    otf = OnTheFlyTokenizer(
        cfg=_make_cfg(hf, tcfg, seq_len=32),
        split=SplitConfig(),
        tokenizer=tokenizer,
    )
    batches = list(otf.iter_dict_batches(yield_rows_per_batch=4))
    assert len(batches) == 1
    b = batches[0]
    assert b["input_ids"].dtype == torch.int64
    assert b["attention_mask"].dtype == torch.bool
    assert b["input_ids"].shape == (2, 32)
    assert b["attention_mask"].shape == (2, 32)
    # Mask should turn off beyond the real content length.
    assert (b["attention_mask"].sum(dim=-1) <= 32).all()
    assert (b["attention_mask"].sum(dim=-1) > 0).all()


def test_strip_historical_thinking_keeps_last_only():
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "a1",
            "thinking": "historical thought — should be dropped",
        },
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "a2 (final)",
            "thinking": "final thought — keep",
        },
    ]
    cleaned = _strip_historical_thinking(messages)
    assert cleaned[1].get("thinking") is None
    assert cleaned[3]["thinking"] == "final thought — keep"


def test_strip_historical_thinking_handles_list_content():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "drop me"},
                {"type": "text", "text": "keep me"},
            ],
        },
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "final-thought-keep"},
                {"type": "text", "text": "final-answer"},
            ],
        },
    ]
    cleaned = _strip_historical_thinking(messages)
    types_first = {c.get("type") for c in cleaned[0]["content"]}
    assert types_first == {"text"}
    types_last = {c.get("type") for c in cleaned[2]["content"]}
    assert types_last == {"thinking", "text"}


# Gemma-4 style chat template: same structure as google/gemma-*-it (role
# mapping, <start_of_turn>/<end_of_turn> delimiters, add_generation_prompt
# hook, optional enable_thinking channel).
GEMMA_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + (message['role'] if message['role'] != 'assistant' else 'model') + '\n' }}"
    "{{ message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n"
    "{% if enable_thinking %}<think>\n{% endif %}"
    "{% endif %}"
)


def test_gemma_style_chat_template_roundtrip(gpt2_tokenizer):
    """Verify our conversation-mode tokenization produces the Gemma control
    structure (role mapping, turn delimiters) when the tokenizer's chat
    template matches Gemma's."""
    tokenizer = gpt2_tokenizer
    tokenizer.chat_template = GEMMA_STYLE_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    convos = [
        [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "four"},
        ]
    ]
    hf = datasets.Dataset.from_dict({"messages": convos})
    tcfg = TokenizationConfig(
        mode=TokenizationMode.CONVERSATION,
        packing=PackingMode.PAD,
        messages_column="messages",
        # No generation prompt — we're tokenizing finished assistant turns.
        chat_template_kwargs={"add_generation_prompt": False},
    )
    otf = OnTheFlyTokenizer(
        cfg=_make_cfg(hf, tcfg, seq_len=64),
        split=SplitConfig(),
        tokenizer=tokenizer,
    )
    (batch,) = list(otf.iter_dict_batches(yield_rows_per_batch=4))
    real_len = int(batch["attention_mask"][0].sum())
    decoded = tokenizer.decode(batch["input_ids"][0, :real_len])

    # Roles: Gemma calls the assistant "model" in its template — this is the
    # rule we need to preserve for Gemma-4 correctness.
    assert "<start_of_turn>user" in decoded
    assert "<start_of_turn>model" in decoded
    assert "<start_of_turn>assistant" not in decoded
    assert decoded.count("<end_of_turn>") == 2
    assert "what is 2+2?" in decoded
    assert "four" in decoded


def test_chat_template_kwargs_passed_through_enable_thinking(gpt2_tokenizer):
    """Sanity-check that `enable_thinking` actually reaches the template, by
    using a template that conditionally emits a <think> block when it's set."""
    tokenizer = gpt2_tokenizer
    tokenizer.chat_template = GEMMA_STYLE_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    convos = [[{"role": "user", "content": "hello"}]]
    hf = datasets.Dataset.from_dict({"messages": convos})

    def decode_one(chat_kwargs: dict):
        tcfg = TokenizationConfig(
            mode=TokenizationMode.CONVERSATION,
            packing=PackingMode.PAD,
            messages_column="messages",
            chat_template_kwargs=chat_kwargs,
        )
        otf = OnTheFlyTokenizer(
            cfg=_make_cfg(hf, tcfg, seq_len=64),
            split=SplitConfig(),
            tokenizer=tokenizer,
        )
        (batch,) = list(otf.iter_dict_batches(yield_rows_per_batch=4))
        real_len = int(batch["attention_mask"][0].sum())
        return tokenizer.decode(batch["input_ids"][0, :real_len])

    with_think = decode_one(
        {"add_generation_prompt": True, "enable_thinking": True}
    )
    without_think = decode_one(
        {"add_generation_prompt": True, "enable_thinking": False}
    )
    assert "<think>" in with_think
    assert "<think>" not in without_think


def test_min_seq_len_drops_short_docs(gpt2_tokenizer):
    """Newly-renamed min_seq_len should drop tokenized docs below the
    threshold. PACK with min_seq_len=0 keeps everything; with a high
    threshold, short rows disappear."""
    texts = ["a", "b", "c"]  # each will tokenize to ~1-2 tokens
    hf = datasets.Dataset.from_dict({"text": texts})

    def run(min_seq_len: int):
        tcfg = TokenizationConfig(
            mode=TokenizationMode.RAW_TEXT,
            packing=PackingMode.PACK,
            map_num_proc=None,
            min_seq_len=min_seq_len,
        )
        otf = OnTheFlyTokenizer(
            cfg=_make_cfg(hf, tcfg, seq_len=16),
            split=SplitConfig(),
            tokenizer=gpt2_tokenizer,
        )
        total = sum(t.numel() for t in otf.iter_tensor_batches(yield_rows_per_batch=4))
        # Also access the doc iterator directly to count kept docs cleanly.
        kept = sum(1 for _ in otf._iter_doc_ids())
        return total, kept

    _, kept_all = run(min_seq_len=0)
    _, kept_filtered = run(min_seq_len=100)
    assert kept_all == len(texts)
    assert kept_filtered == 0
