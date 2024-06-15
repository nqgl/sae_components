CE_STR = "Exception Cache Location Info: Last tracked subcache was "


def locate_cache_exception(e, cache, name=None):
    if name is None:
        name = ""
    else:
        name = f".{name}"
    if not any([a.startswith(CE_STR) for a in e.args]):
        e.args = (f"{CE_STR}{cache._name}{name}",) + e.args
    return e
