"""Tests for ``standardize`` -- the comparable-lambda key (glen-inbox item 1).

Run:  PYTHONPATH=research/src python -m saeco_research.cgraph2.funcstd_demo
"""

import math

from saeco_research.cgraph2.funcstd import standardize

zzz = math.log


def f(x):
    return x + 1


def g(x):
    return math.log(x + 1)


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def eq(a, b):
    return standardize(a) == standardize(b)


def main():
    # same source twice
    ok("same source lambda", eq(lambda x: x + 1, lambda x: x + 1))

    # alpha-equivalence (parameter rename)
    ok("alpha-rename: lambda y vs lambda x", eq(lambda y: y + 1, lambda x: x + 1))

    # a named def with the same body as a lambda
    ok("def f vs lambda (same body)", eq(f, lambda x: x + 1))

    # nested call to an analyzable function is inlined (beta reduction)
    ok("inline: lambda y: f(y) == lambda x: x + 1", eq(lambda y: f(y), lambda x: x + 1))

    # external calls through aliases unify (zzz is math.log)
    ok(
        "alias: math.log vs zzz",
        eq(lambda x: math.log(x + 1), lambda z: zzz(z + 1)),
    )
    ok("inline external-call fn g vs inline lambda", eq(g, lambda x: math.log(x + 1)))

    # --- it must NOT collapse genuinely different functions ---
    ok("x+1 != x+2", not eq(lambda x: x + 1, lambda x: x + 2))
    ok("x+1 != x*1", not eq(lambda x: x + 1, lambda x: x * 1))
    ok("log != exp", not eq(lambda x: math.log(x), lambda x: math.exp(x)))
    ok("operand order matters: x-1 != 1-x", not eq(lambda x: x - 1, lambda x: 1 - x))

    # closures over distinct captured constants stay distinct
    def adder(k):
        return lambda x: x + k

    ok("closure k=1 == k=1", eq(adder(1), adder(1)))
    ok("closure k=1 != k=2", not eq(adder(1), adder(2)))

    # un-analyzable callables (builtins / C) fall back to identity -- still
    # self-equal, and equal across references to the SAME object
    ok("builtin self-equal", eq(math.log, math.log))
    ok("builtin alias zzz is math.log", eq(zzz, math.log))
    ok("distinct builtins differ", not eq(math.log, math.exp))

    # the payoff: wired into op() consing, equivalent inline lambdas fuse to the
    # SAME interned node -- no need to hoist them to a shared named function
    from saeco_research.cgraph2 import Input, op

    fused = op(lambda x: x + 1)(Input) is op(lambda y: y + 1)(Input)
    ok("op(lambda x: x+1) fuses with op(lambda y: y+1)", fused)
    distinct = op(lambda x: x + 1)(Input) is not op(lambda x: x + 2)(Input)
    ok("op(...+1) does NOT fuse with op(...+2)", distinct)

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
