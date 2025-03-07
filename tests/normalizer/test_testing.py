from hypothesis import example, given, strategies as st


@given(st.text())
def test_text_examples(text: str):
    assert text[:2] != "ab"
