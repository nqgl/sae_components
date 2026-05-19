from saeco.data.config.tokenization_config import (
    PackingMode,
    TokenizationConfig,
    TokenizationMode,
)


def test_idstr_fragment_empty_for_pretokenized():
    cfg = TokenizationConfig()
    assert cfg.mode == TokenizationMode.PRETOKENIZED
    assert cfg.idstr_fragment() == ""


def test_idstr_fragment_includes_mode_and_packing():
    cfg = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT,
        packing=PackingMode.PACK,
    )
    frag = cfg.idstr_fragment()
    assert "raw_text" in frag
    assert "pack" in frag


def test_idstr_fragment_differs_on_chat_template_kwargs():
    a = TokenizationConfig(
        mode=TokenizationMode.CONVERSATION,
        packing=PackingMode.PAD,
        chat_template_kwargs={"enable_thinking": False},
    )
    b = TokenizationConfig(
        mode=TokenizationMode.CONVERSATION,
        packing=PackingMode.PAD,
        chat_template_kwargs={"enable_thinking": True},
    )
    assert a.idstr_fragment() != b.idstr_fragment()


def test_idstr_fragment_stable_for_matching_settings():
    a = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT,
        tokenizer_kwargs={"add_special_tokens": True},
    )
    b = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT,
        tokenizer_kwargs={"add_special_tokens": True},
    )
    assert a.idstr_fragment() == b.idstr_fragment()


def test_template_version_tag_invalidates():
    base = TokenizationConfig(mode=TokenizationMode.RAW_TEXT)
    bumped = TokenizationConfig(
        mode=TokenizationMode.RAW_TEXT, template_version_tag="v2"
    )
    assert base.idstr_fragment() != bumped.idstr_fragment()
    assert "tv_v2" in bumped.idstr_fragment()
