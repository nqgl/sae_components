import functools
from typing_extensions import deprecated

DEBUGGING = False
DEBUG_INDENT = 0
GETTING = False
DB_BUF = ""
TRACE = []


def dbprint(*a):
    global DB_BUF
    DB_BUF += " ".join(map(str, a)) + "\n"


def debug_enter(self, field, current):
    global DEBUGGING
    global DEBUG_INDENT
    global GETTING
    global TRACE
    if DEBUGGING:
        nice_name = field[1:]
        IND = "   " * DEBUG_INDENT
        prevget = GETTING
        TRACE.append(nice_name)
        if current is None and not GETTING:
            dbprint(" -> ".join(TRACE))
            GETTING = True
        if GETTING:
            if current is None:
                dbprint(f"{IND}GET: {nice_name}")
            else:
                dbprint(f"{IND}HAS: {nice_name}")

        def cleanup():
            global GETTING
            global DEBUG_INDENT
            global DB_BUF
            TRACE.pop()
            DEBUG_INDENT -= 1
            GETTING = prevget
            if not GETTING:
                print(DB_BUF)
                DB_BUF = ""

        DEBUG_INDENT += 1
        return cleanup
    return lambda: None


def defer_to_and_set(_field):
    def wrapper(fn):

        @functools.wraps(fn)
        def inner(self):
            if isinstance(_field, str):
                field = _field
            else:
                field = _field()
                assert isinstance(field, str)
            current = getattr(self, field, None)
            debug_exit = debug_enter(self, field, current)
            if current is None:
                setattr(self, field, fn(self))
                current = getattr(self, field)
                assert current is not None
            debug_exit()
            return current

        return inner

    return wrapper


def lazycall(mth, name=None):
    if isinstance(mth, str):
        return lambda x: lazycall(x, name=mth)
    name = name or f"_{mth.__name__}"
    return defer_to_and_set(name)(mth)
