from __future__ import annotations

import pytest
import torch
from torch import Tensor

# Adjust this import to your actual location.
# If DictBatch lives in the same file during bring-up, you can also import it directly.
from saeco.data.dict_batch.dict_batch import DictBatch, dictbatch_field

# ---------------------------
# Test fixtures / test classes
# ---------------------------


@DictBatch.auto_other_fields
class MyBatch(DictBatch):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    prompt: str
    doc_id: int


def ones_maker(batch_size: int) -> Tensor:
    return torch.ones(batch_size, 8, dtype=torch.long)


@DictBatch.auto_other_fields
class WithOptionalTensor(DictBatch):
    a: Tensor = dictbatch_field(factory=ones_maker)
    b: Tensor | None
    c: Tensor | None = None


@pytest.fixture
def mybatch() -> MyBatch:
    return MyBatch(
        {
            "input_ids": torch.arange(12, dtype=torch.long).reshape(3, 4),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "labels": torch.zeros(3, 4, dtype=torch.long),
        },
        prompt="hi",
        doc_id=7,
    )


# ---------------------------
# Core DictBatch behavior
# ---------------------------


def test_auto_other_fields_populates_fields() -> None:
    assert set(MyBatch.TENSOR_DATA_FIELDS) >= {"input_ids", "attention_mask", "labels"}
    assert set(MyBatch.OTHER_DATA_FIELDS) >= {"prompt", "doc_id"}


def test_ctor_rejects_mixed_data_and_tensor_kwargs() -> None:
    with pytest.raises(
        ValueError, match="either via the data mapping OR as keyword arguments"
    ):
        MyBatch(
            {
                "input_ids": torch.ones(2, 3),
                "attention_mask": torch.ones(2, 3),
                "labels": torch.ones(2, 3),
            },
            input_ids=torch.ones(2, 3),
            prompt="x",
            doc_id=1,
        )


def test_ctor_checks_batch_size_consistency() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        MyBatch(
            {
                "input_ids": torch.ones(2, 3, dtype=torch.long),
                "attention_mask": torch.ones(3, 3, dtype=torch.long),  # mismatch
                "labels": torch.ones(2, 3, dtype=torch.long),
            },
            prompt="x",
            doc_id=1,
        )


def test_missing_required_tensor_field_errors() -> None:
    with pytest.raises(ValueError, match="Missing required tensor field 'labels'"):
        MyBatch(
            {
                "input_ids": torch.ones(2, 3, dtype=torch.long),
                "attention_mask": torch.ones(2, 3, dtype=torch.long),
                # labels missing
            },
            prompt="x",
            doc_id=1,
        )


def test_optional_tensor_fields_must_exist_but_can_be_none() -> None:
    # b is required to be present (OPTIONAL_TENSOR_FIELDS), but can be None
    ok = WithOptionalTensor(a=torch.ones(3, 8, dtype=torch.long), b=None)
    assert ok["b"] is None

    # Missing b should error
    with pytest.raises(ValueError, match="Missing required tensor field 'b'"):
        WithOptionalTensor(a=torch.ones(3, 8, dtype=torch.long))


def test_default_factory_renders_from_batch_size() -> None:
    b = WithOptionalTensor(b=torch.ones(4, 8, dtype=torch.long))
    assert b["a"].shape == (4, 8)
    assert torch.all(b["a"] == 1)


def test_attribute_access_and_setattr_for_tensor_fields(mybatch: MyBatch) -> None:
    assert isinstance(mybatch.input_ids, Tensor)
    assert torch.equal(mybatch["input_ids"], mybatch.input_ids)

    mybatch.input_ids = torch.zeros_like(mybatch.input_ids)
    assert torch.all(mybatch["input_ids"] == 0)


def test_getitem_str_returns_tensor(mybatch: MyBatch) -> None:
    t = mybatch["labels"]
    assert isinstance(t, Tensor)
    assert t.shape == (3, 4)


def test_getitem_slice_returns_same_type_and_slices_all_tensors(
    mybatch: MyBatch,
) -> None:
    b2 = mybatch[:2]
    assert isinstance(b2, MyBatch)
    assert b2.batch_size == 2
    assert b2.input_ids.shape[0] == 2
    assert b2.attention_mask.shape[0] == 2
    assert b2.labels.shape[0] == 2
    # extras copied
    assert b2.prompt == mybatch.prompt
    assert b2.doc_id == mybatch.doc_id


def test_apply_func_applies_over_present_tensors_and_preserves_none() -> None:
    b = WithOptionalTensor(
        a=torch.ones(2, 8, dtype=torch.long),
        b=None,
        c=torch.ones(2, 8, dtype=torch.long),
    )
    out = b.apply_func(lambda t: t + 2)
    assert torch.all(out["a"] == 3)
    assert out["b"] is None
    assert torch.all(out["c"] == 3)


def test_present_keys_skips_none() -> None:
    b = WithOptionalTensor(
        a=torch.ones(2, 8, dtype=torch.long),
        b=None,
        c=torch.ones(2, 8, dtype=torch.long),
    )
    assert b.present_keys() == {"a", "c"}


def test_clone_produces_distinct_tensors(mybatch: MyBatch) -> None:
    c = mybatch.clone()
    assert isinstance(c, MyBatch)
    assert c is not mybatch
    assert c.input_ids is not mybatch.input_ids
    assert torch.equal(c.input_ids, mybatch.input_ids)


def test_contiguous_keeps_values_equal(mybatch: MyBatch) -> None:
    # Make a non-contiguous view
    nb = mybatch.apply_func(lambda t: t[:, ::2])
    assert not nb.input_ids.is_contiguous()
    cb = nb.contiguous()
    assert cb.input_ids.is_contiguous()
    assert torch.equal(cb.input_ids, nb.input_ids)


def test_reshape_works(mybatch: MyBatch) -> None:
    r = mybatch.reshape(3, 2, 2)
    assert r.input_ids.shape == (3, 2, 2)
    assert r.attention_mask.shape == (3, 2, 2)
    assert r.labels.shape == (3, 2, 2)


def test_gather_works() -> None:
    b = MyBatch(
        {
            "input_ids": torch.tensor([[10, 11, 12], [20, 21, 22]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.tensor([[0, 1, 2], [3, 4, 5]]),
        },
        prompt="p",
        doc_id=1,
    )
    idx = torch.tensor([[2, 1], [0, 2]])
    out = b.gather(1, idx)
    assert torch.equal(out.input_ids, torch.tensor([[12, 11], [20, 22]]))
    assert torch.equal(out.labels, torch.tensor([[2, 1], [3, 5]]))


def test_shapes_and_shape_helpers(mybatch: MyBatch) -> None:
    shapes = mybatch.shapes
    assert shapes["input_ids"] == (3, 4)
    assert mybatch.shape[0] == 3
    assert mybatch.shape["labels"] == (3, 4)
    assert mybatch.shape[1] == {"input_ids": 4, "attention_mask": 4, "labels": 4}


def test_updated_with_preserves_other_data(mybatch: MyBatch) -> None:
    out = mybatch.updated_with(labels=torch.ones_like(mybatch.labels))
    assert isinstance(out, MyBatch)
    assert out.prompt == mybatch.prompt
    assert out.doc_id == mybatch.doc_id
    assert torch.all(out.labels == 1)


def test_cast_convert_identity_if_already_exact_type(mybatch: MyBatch) -> None:
    out = MyBatch.cast_convert(mybatch)
    assert out is mybatch


def test_cast_convert_can_add_tensor_fields_and_other_fields(mybatch: MyBatch) -> None:
    # Convert base DictBatch -> MyBatch while supplying required extras/fields
    base = DictBatch(
        {
            "input_ids": mybatch.input_ids.clone(),
            "attention_mask": mybatch.attention_mask.clone(),
            "labels": mybatch.labels.clone(),
        }
    )
    out = MyBatch.cast_convert(base, prompt="x", doc_id=123)
    assert isinstance(out, MyBatch)
    assert out.prompt == "x"
    assert out.doc_id == 123


def test_split_produces_correct_number_and_sizes(mybatch: MyBatch) -> None:
    parts = mybatch.split([1, 2], dim=0)
    assert [p.batch_size for p in parts] == [1, 2]
    assert all(isinstance(p, MyBatch) for p in parts)


def test_set_split_and_recombine_roundtrip(mybatch: MyBatch) -> None:
    split = mybatch.set_split({"input_ids"})
    assert set(split.a.keys()) == {"input_ids"}
    assert set(split.b.keys()) == {"attention_mask", "labels"}
    recombined = split.recombine()
    assert isinstance(
        recombined, DictBatch
    )  # recombine uses cls stored in SplitDictBatch; your current code passes self.__class__
    # If you want it to be MyBatch, you’ll adjust set_split() to preserve subclass; test below is future-facing:
    # assert isinstance(recombined, MyBatch)


def test_cat_list_and_stack_list_handle_optional_none_correctly() -> None:
    # This test will catch the common bug: torch.cat(... if cond else None, dim=...) vs (torch.cat(...) if cond else None)
    b1 = WithOptionalTensor(
        a=torch.ones(2, 8, dtype=torch.long),
        b=None,
        c=torch.ones(2, 8, dtype=torch.long),
    )
    b2 = WithOptionalTensor(
        a=torch.ones(3, 8, dtype=torch.long),
        b=None,
        c=torch.ones(3, 8, dtype=torch.long),
    )

    cat = WithOptionalTensor.cat_list([b1, b2], dim=0)
    assert cat.batch_size == 5
    assert cat["b"] is None
    assert cat["a"].shape == (5, 8)
    assert cat["c"].shape == (5, 8)

    b1 = WithOptionalTensor(
        a=torch.ones(3, 8, dtype=torch.long),
        b=None,
        c=torch.ones(3, 8, dtype=torch.long),
    )
    b2 = WithOptionalTensor(
        a=torch.ones(3, 8, dtype=torch.long),
        b=None,
        c=torch.ones(3, 8, dtype=torch.long),
    )

    stk = WithOptionalTensor.stack_list([b1, b2], dim=0)
    assert stk["b"] is None
    assert stk.a.shape == (2, 3, 8)


# ---------------------------
# Nesting: future-facing tests
# ---------------------------
# These describe desired behavior once you allow DictBatches inside DictBatches.
# Flip xfail -> normal as you implement nesting.


def _make_child(batch_size: int, offset: int) -> DictBatch:
    return DictBatch(
        {
            "x": torch.arange(
                offset, offset + batch_size * 2, dtype=torch.long
            ).reshape(batch_size, 2),
            "y": torch.ones(batch_size, 2, dtype=torch.long),
        }
    )


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_ctor_allows_child_dictbatch_value() -> None:
    child = _make_child(3, 0)
    parent = DictBatch({"a": torch.zeros(3, 1), "child": child})
    assert isinstance(parent["child"], DictBatch)
    assert parent.batch_size == 3


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_batch_size_must_match() -> None:
    child = _make_child(2, 0)
    with pytest.raises(ValueError, match="batch size"):
        DictBatch({"a": torch.zeros(3, 1), "child": child})


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_indexing_indexes_child_tensors_too() -> None:
    child = _make_child(3, 0)
    parent = DictBatch({"a": torch.arange(3).reshape(3, 1), "child": child})

    out = parent[:2]
    assert out["a"].shape[0] == 2
    assert out["child"]["x"].shape[0] == 2
    assert torch.equal(out["child"]["x"], child["x"][:2])


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_apply_func_recurses_into_child() -> None:
    child = _make_child(3, 0)
    parent = DictBatch({"a": torch.ones(3, 1, dtype=torch.long), "child": child})

    out = parent.apply_func(lambda t: t + 5)
    assert torch.all(out["a"] == 6)
    assert torch.all(out["child"]["x"] == child["x"] + 5)
    assert torch.all(out["child"]["y"] == 6)


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_cat_list_concats_children_via_child_cat_list() -> None:
    p1 = DictBatch({"a": torch.zeros(2, 1), "child": _make_child(2, 0)})
    p2 = DictBatch({"a": torch.ones(3, 1), "child": _make_child(3, 100)})

    out = DictBatch.cat_list([p1, p2], dim=0)
    assert out.batch_size == 5
    assert isinstance(out["child"], DictBatch)
    assert out["child"]["x"].shape[0] == 5
    assert torch.equal(out["a"][:2], p1["a"])
    assert torch.equal(out["a"][2:], p2["a"])


# @pytest.mark.xfail(reason="Nested DictBatch values not supported yet")
def test_nested_present_keys_counts_child_as_present() -> None:
    child = _make_child(3, 0)
    parent = DictBatch({"a": torch.zeros(3, 1), "child": child, "maybe": None})
    assert parent.present_keys() == {"a", "child"}


if __name__ == "__main__":
    test_nested_indexing_indexes_child_tensors_too()
    test_cast_convert_can_add_tensor_fields_and_other_fields(
        mybatch=MyBatch(
            {
                "input_ids": torch.arange(12, dtype=torch.long).reshape(3, 4),
                "attention_mask": torch.ones(3, 4, dtype=torch.long),
                "labels": torch.zeros(3, 4, dtype=torch.long),
            },
            prompt="hi",
            doc_id=7,
        )
    )
    test_nested_apply_func_recurses_into_child()
