def getsite(obj, name):
    if "." in name:
        name = name.split(".")
        for n in name:
            obj = getsite(obj, n)
        return obj
    if name.isnumeric():
        return obj[int(name)]
    return getattr(obj, name)


def setsite(obj, name, value):
    if "." in name:
        sp = name.split(".")
        end = sp.pop()
        name = ".".join(sp)
        obj = getsite(obj, name)
    setattr(obj, end, value)


translations = {
    "blocks.": "h.",
    "hook_resid_pre": "input",
    "hook_resid_post": "output.0",
    "hook_resid_mid": "ln_2.input",
}


def tlsite_to_nnsite(tl_name):
    while any([k in tl_name for k in translations.keys()]):
        for k, v in translations.items():
            tl_name = tl_name.replace(k, v)
    return f"transformer.{tl_name}"
