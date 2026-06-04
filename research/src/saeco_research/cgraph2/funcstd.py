"""``standardize(fn)`` -- a structural, comparable key for a callable.

The interning core cons-keys producers on identity, so to share a node between
two *equivalent* inline functions you'd otherwise have to hoist them to one
named module-level function. ``standardize`` removes that chore: it decompiles a
straight-line expression function into a canonical expression tree, so

    standardize(lambda x: x + 1) == standardize(lambda y: y + 1)   # alpha-rename
    standardize(def f(x): return x + 1) == standardize(lambda x: x + 1)
    standardize(lambda x: math.log(x + 1)) == standardize(lambda z: zzz(z + 1))

all hold (the last because ``zzz is math.log`` -- names resolve to the object).
Calls to other analyzable functions are *inlined* (beta-reduced), so

    def f(x): return x + 1
    standardize(lambda y: f(y)) == standardize(lambda x: x + 1)

The contract is one-directional and safety-first: equivalent functions *may*
share a key, but two functions never share a key unless their canonical
expression trees are identical (i.e. they compute the same thing). Anything not
analyzable as a pure straight-line expression -- branches, loops, locals,
builtins/C functions, unhashable constants -- falls back to ``id`` identity. A
missed match costs a missed fusion; it is never a wrong fusion.

Only the opcodes a pure arithmetic/`call` expression emits are modeled; every
other opcode bails to the identity fallback.
"""

import builtins
import dis


class _BailError(Exception):
    """Raised when the function isn't a pure straight-line expression we model;
    caught by :func:`_try_expr_tree`, which then returns ``None`` (id fallback)."""


_NULL = ("\0null",)  # sentinel for a self/NULL slot on the symbolic stack


def _obj_key(obj):
    """A stable identity for a referenced (non-inlined) object: its
    module+qualname when available (so the same builtin reached via different
    aliases keys alike), else its ``id``."""
    mod = getattr(obj, "__module__", None)
    qual = getattr(obj, "__qualname__", None)
    if mod is not None and qual is not None:
        return ("q", mod, qual)
    return ("id", id(obj))


def _resolve_global(fn, name):
    if name in fn.__globals__:
        return fn.__globals__[name]
    return getattr(builtins, name)


def _resolve_deref(fn, name):
    code = fn.__code__
    names = code.co_freevars + code.co_cellvars
    idx = names.index(name)
    return fn.__closure__[idx].cell_contents


def _is_analyzable(obj):
    return hasattr(obj, "__code__") and hasattr(obj, "__globals__")


def _subst(tree, args):
    """Replace ``("arg", i)`` leaves in an inlined callee's tree with the
    caller's argument subtrees (beta reduction)."""
    if isinstance(tree, tuple) and tree and tree[0] == "arg":
        return args[tree[1]]
    if isinstance(tree, tuple):
        return tuple(_subst(t, args) for t in tree)
    return tree


def _expr_tree(fn, seen):
    code = fn.__code__
    if code.co_flags & (0x04 | 0x08):  # *args / **kwargs -> not a fixed-arity expr
        raise _BailError
    argcount = code.co_argcount
    if code.co_kwonlyargcount:
        raise _BailError
    stack: list = []
    result = None
    for ins in dis.get_instructions(code):
        op = ins.opname
        if op in ("RESUME", "NOP", "CACHE", "PRECALL", "COPY_FREE_VARS", "MAKE_CELL"):
            continue
        elif op == "LOAD_FAST":
            if ins.arg >= argcount:
                raise _BailError  # a stored local; not a pure expression
            stack.append(("arg", ins.arg))
        elif op == "LOAD_CONST":
            v = ins.argval
            if isinstance(v, type(code)):  # nested code object
                raise _BailError
            stack.append(("const", v))
        elif op == "PUSH_NULL":
            stack.append(_NULL)
        elif op == "LOAD_GLOBAL":
            if ins.arg & 1:  # "+ NULL" -> a self slot, pushed under the global
                stack.append(_NULL)
            stack.append(("ref", _resolve_global(fn, ins.argval)))
        elif op == "LOAD_DEREF":
            stack.append(("ref", _resolve_deref(fn, ins.argval)))
        elif op == "LOAD_ATTR":
            if ins.arg & 1:  # method-style load -> bound self; out of scope
                raise _BailError
            base = stack.pop()
            if not (isinstance(base, tuple) and base[0] == "ref"):
                raise _BailError
            stack.append(("ref", getattr(base[1], ins.argval)))
        elif op == "BINARY_OP":
            b, a = stack.pop(), stack.pop()
            stack.append(("binop", ins.argrepr, a, b))
        elif op == "BINARY_SUBSCR":
            b, a = stack.pop(), stack.pop()
            stack.append(("subscr", a, b))
        elif op in ("UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT"):
            stack.append(("unary", op, stack.pop()))
        elif op == "COMPARE_OP":
            b, a = stack.pop(), stack.pop()
            stack.append(("cmp", ins.argrepr, a, b))
        elif op == "CALL":
            args = [stack.pop() for _ in range(ins.arg)][::-1]
            x, y = stack.pop(), stack.pop()
            if x == _NULL:
                callee = y
            elif y == _NULL:
                callee = x
            else:
                raise _BailError  # bound-method call (real self) -> out of scope
            stack.append(_make_call(callee, args, seen))
        elif op == "RETURN_VALUE":
            result = stack.pop()
        elif op == "RETURN_CONST":
            result = ("const", ins.argval)
        else:
            raise _BailError
    if result is None or stack:
        raise _BailError
    return result


def _make_call(callee, args, seen):
    if isinstance(callee, tuple) and callee[0] == "ref":
        obj = callee[1]
        if _is_analyzable(obj) and id(obj) not in seen:
            inner = _expr_tree(obj, seen | {id(obj)})
            return _subst(inner, args)
        return ("call", _obj_key(obj), tuple(args))
    return ("call", callee, tuple(args))


def _try_expr_tree(fn):
    if not _is_analyzable(fn):
        return None
    try:
        tree = _expr_tree(fn, {id(fn)})
        hash(tree)  # canonical key must be hashable to cons on
    except Exception:  # _BailError or any analysis failure -> identity fallback
        return None
    return tree


def standardize(fn):
    """A hashable canonical key for ``fn`` (see module docstring)."""
    tree = _try_expr_tree(fn)
    if tree is not None:
        return ("expr", tree)
    return ("id", id(fn))
