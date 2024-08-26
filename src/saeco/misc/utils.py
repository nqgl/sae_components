def useif(cond, *args, **kwargs):
    assert args or kwargs and not (args and kwargs)
    if args:
        return args if cond else []
    return kwargs if cond else {}
