import functools
import itertools
import math
from typing import NamedTuple

import pytest

from saeco_research.cgraph2.funcstd import standardize

LOG_ALIAS = math.log
MATH_ALIAS = math
UNHASHABLE_GLOBAL = [10, 20]


def _key(fn):
    return standardize(fn)


def _assert_id_key(fn):
    key = _key(fn)
    assert key[0] == "id"
    assert key == _key(fn)
    hash(key)
    return key


def _same_key(left, right):
    return _key(left) == _key(right)


def _add_one_def(x):
    return x + 1


def _inline_add_one(x):
    return x + 1


def _defaulted_increment(x=10):
    return x + 1


def _default_two(y=2):
    return y + 1


def _default_three(y=3):
    return y + 1


def _ignores_extra_positional(x, y):
    return x + 1


def _capture_add(value):
    def inner(x):
        return x + value

    return inner


def _capture_return(value):
    def inner():
        return value

    return inner


def _branch(x):
    if x:
        return x + 1
    return x - 1


def _loop(x):
    for item in range(2):
        x += item
    return x


def _local_assignment(x):
    y = x + 1
    return y


def _walrus(x):
    return (y := x + 1) + y


def _list_comprehension(x):
    return [item + x for item in range(2)]


def _dict_comprehension(x):
    return {item: item + x for item in range(2)}


def _set_comprehension(x):
    return {item + x for item in range(2)}


def _generator_expression(x):
    return sum(item + x for item in range(2))


def _tuple_literal(x):
    return (x, 1)


def _list_literal(x):
    return [x, 1]


def _dict_literal(x):
    return {"x": x}


def _set_literal(x):
    return {x, 1}


def _f_string(x):
    return f"{x}"


def _boolean_and(x):
    return x and 1


def _boolean_or(x):
    return x or 1


def _ternary(x):
    return 1 if x else 2


def _starred_call(values):
    return max(*values)


def _keyword_call(x):
    return int(x, base=10)


def _varargs(*args):
    return args[0]


def _kwargs(**kwargs):
    return kwargs["x"]


def _kwonly(x, *, scale=1):
    return x + scale


def _unhashable_global():
    return UNHASHABLE_GLOBAL


def _unhashable_global_subscript(index):
    return UNHASHABLE_GLOBAL[index]


def _undersupplied_defaulted_callee():
    return _defaulted_increment()


def _recursive(x):
    return _recursive(x)


def _mutual_a(x):
    return _mutual_b(x)


def _mutual_b(x):
    return _mutual_a(x)


class _CallableObject:
    def __call__(self, x):
        return x + 1


def test_positive_fusions_cover_supported_expression_shapes() -> None:
    assert _same_key(lambda x: x + 1, lambda y: y + 1)
    assert _same_key(_add_one_def, lambda x: x + 1)
    assert _same_key(lambda x: LOG_ALIAS(x + 1), lambda z: math.log(z + 1))
    assert _same_key(lambda y: _inline_add_one(y), lambda x: x + 1)
    assert _same_key(lambda y: _defaulted_increment(y), lambda x: x + 1)

    assert _same_key(lambda x: -x, lambda y: -y)
    assert _same_key(lambda x: x < 3, lambda y: y < 3)
    assert _same_key(lambda data: data["b"], lambda item: item["b"])
    assert _same_key(lambda x: math.tau + x, lambda y: MATH_ALIAS.tau + y)
    assert _same_key(
        lambda x: math.log(abs(x + 1)),
        lambda y: LOG_ALIAS(abs(y + 1)),
    )


def test_behaviorally_distinct_functions_have_distinct_keys() -> None:
    distinct_pairs = [
        (lambda x: x + 1, lambda x: x + 2),
        (lambda x: x + 1, lambda x: x * 1),
        (lambda x: x - 1, lambda x: 1 - x),
        (lambda x: math.log(x), lambda x: math.exp(x)),
        (lambda data: data["a"], lambda data: data["b"]),
    ]
    for left, right in distinct_pairs:
        assert not _same_key(left, right)

    assert _assert_id_key(_default_two)[0] == "id"
    assert _assert_id_key(_default_three)[0] == "id"
    assert not _same_key(_default_two, _default_three)
    assert not _same_key(lambda y=2: y + 1, lambda y=3: y + 1)
    assert _default_two() != _default_three()

    assert _assert_id_key(_ignores_extra_positional)[0] == "id"
    assert not _same_key(lambda x: x + 1, _ignores_extra_positional)


def test_closure_references_track_same_and_different_captured_objects() -> None:
    assert _same_key(_capture_add(1), _capture_add(1))
    assert not _same_key(_capture_add(1), _capture_add(2))

    marker = object()
    assert _same_key(_capture_return(marker), _capture_return(marker))
    assert not _same_key(_capture_return(object()), _capture_return(object()))


_SAFE_BAIL_CASES = [
    ("branch", _branch),
    ("loop", _loop),
    ("local-assignment", _local_assignment),
    ("walrus", _walrus),
    ("list-comprehension", _list_comprehension),
    ("dict-comprehension", _dict_comprehension),
    ("set-comprehension", _set_comprehension),
    ("generator-expression", _generator_expression),
    ("tuple-literal", _tuple_literal),
    ("list-literal", _list_literal),
    ("dict-literal", _dict_literal),
    ("set-literal", _set_literal),
    ("f-string", _f_string),
    ("boolean-and", _boolean_and),
    ("boolean-or", _boolean_or),
    ("ternary", _ternary),
    ("starred-call", _starred_call),
    ("keyword-call", _keyword_call),
    ("varargs", _varargs),
    ("kwargs", _kwargs),
    ("kwonly", _kwonly),
    ("unhashable-global", _unhashable_global),
    ("unhashable-global-subscript", _unhashable_global_subscript),
    ("undersupplied-defaulted-callee", _undersupplied_defaulted_callee),
    ("recursive", _recursive),
    ("mutual-recursive", _mutual_a),
    ("len", len),
    ("print", print),
    ("partial", functools.partial(pow, 2)),
    ("callable-object", _CallableObject()),
]


@pytest.mark.parametrize(("name", "fn"), _SAFE_BAIL_CASES)
def test_non_analyzable_callables_fall_back_to_id_without_raising(name, fn) -> None:
    assert name
    _assert_id_key(fn)


class _ContractCase(NamedTuple):
    name: str
    fn: object
    reference: object
    samples: tuple[tuple, ...]


_NUMERIC_SAMPLES = ((-2,), (0,), (3,), (4.5,))
_POSITIVE_SAMPLES = ((0.25,), (1.0,), (3.0,))
_ZERO_ARG_SAMPLES = ((),)

_CONTRACT_CASES = [
    _ContractCase("add-one-lambda", lambda x: x + 1, lambda x: x + 1, _NUMERIC_SAMPLES),
    _ContractCase("add-one-def", _add_one_def, lambda x: x + 1, _NUMERIC_SAMPLES),
    _ContractCase("add-two", lambda x: x + 2, lambda x: x + 2, _NUMERIC_SAMPLES),
    _ContractCase("mul-two", lambda x: x * 2, lambda x: x * 2, _NUMERIC_SAMPLES),
    _ContractCase("sub-one", lambda x: x - 1, lambda x: x - 1, _NUMERIC_SAMPLES),
    _ContractCase("one-minus", lambda x: 1 - x, lambda x: 1 - x, _NUMERIC_SAMPLES),
    _ContractCase("negative", lambda x: -x, lambda x: -x, _NUMERIC_SAMPLES),
    _ContractCase(
        "less-than-three",
        lambda x: x < 3,
        lambda x: x < 3,
        _NUMERIC_SAMPLES,
    ),
    _ContractCase(
        "log-alias",
        lambda x: LOG_ALIAS(x + 1),
        lambda x: math.log(x + 1),
        _POSITIVE_SAMPLES,
    ),
    _ContractCase(
        "log-global",
        lambda x: math.log(x + 1),
        lambda x: math.log(x + 1),
        _POSITIVE_SAMPLES,
    ),
    _ContractCase(
        "exp-global",
        lambda x: math.exp(x + 1),
        lambda x: math.exp(x + 1),
        _POSITIVE_SAMPLES,
    ),
    _ContractCase("default-two", _default_two, lambda: 3, _ZERO_ARG_SAMPLES),
    _ContractCase("default-three", _default_three, lambda: 4, _ZERO_ARG_SAMPLES),
]


def test_standardize_equality_implies_same_outputs_on_sampled_contract_cases() -> None:
    for case in _CONTRACT_CASES:
        for args in case.samples:
            assert case.fn(*args) == case.reference(*args)

    for left, right in itertools.combinations(_CONTRACT_CASES, 2):
        if left.samples != right.samples:
            continue
        if _key(left.fn) != _key(right.fn):
            continue
        for args in left.samples:
            assert left.reference(*args) == right.reference(*args), (
                left.name,
                right.name,
                args,
                _key(left.fn),
            )
