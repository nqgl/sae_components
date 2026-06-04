inbox/thoughts dump from me


# comparable lambdas
so we don't have to define simple functions to get consing to work, we should make a function something that can take a lambda and make it into a statically comparible thing.

the simple case is stuff like being able to tell
standardize(lambda x: x + 1) == standardize(lambda x: x + 1)


but we should also be able to tell that eg
standardize(lambda y: y + 1) == standardize(lambda x: x + 1)
or that with
def f(x):
    return x + 1
f is the same as standardize(lambda x: x + 1)

also we should be able to handle nested functions:

def f(x):
    return x + 1
standardize(lambda y: f(y)) == standardize(lambda x: x + 1)

and calls to external functions:
import math
from math import log
zzz = log
standardize(lambda x: math.log(x + 1)) == standardize(lambda z: zzz(z + 1))

> DONE -- `funcstd.standardize` (tests in `funcstd_demo.py`, all cases above pass)
> and it's wired in as the default cons key in `Op.create`, so equivalent inline
> lambdas fuse to one node. It decompiles straight-line *expression* bytecode to
> a canonical expr-tree: alpha-renames args, resolves names to objects (aliases
> unify), and inlines/beta-reduces calls to analyzable fns. Safety-first: it only
> ever matches functions with identical expr-trees (same computation); anything
> non-expression (branches/loops/locals/builtins/unhashable consts) falls back to
> `id` -- a missed fusion, never a wrong one. Not covered yet (deliberately, as
> id-fallback): control flow, statements/multiple-return, method calls with a
> real `self`, commutativity (`a+b` vs `b+a` stay distinct).


# function for the block used by namespace vs class
I think we do need it to be class bc having access to __prepare__ is nice and it's nice to have it be a namespace that's exposed to the outside once instantiate the graph, eg its easy to point to graph.acts or whatever from the outside. But on the other hand it kind of wants to have inputs and return values and be able to express input types and return types which is a little tricky to do
- not actionable right now but wanted to make note and track thoughts on this

> RESOLVED (overnight). `GraphSpec` is the class-block, and it's both the
> namespace *and* the runnable context (D-01): `graph.acts` is exposed and typed
> from outside; inputs are the slots (`op_slot`/`arg`/`sub_slot`, typed via
> `dataclass_transform`) and outputs are the handles (typed `ValueRef`s; a typo
> errors). The "inputs and return types" wish is met by typed slots + typed
> handles; `GraphSpec.of(**handles)` covers the dynamic (non-class-body) case.
