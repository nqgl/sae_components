def useif(cond, **kwargs):
    return kwargs if cond else {}
