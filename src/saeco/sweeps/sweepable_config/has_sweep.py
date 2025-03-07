from saeco.sweeps.sweepable_config.Swept import Swept


from pydantic import BaseModel


def has_sweep(target: BaseModel | dict):
    if isinstance(target, BaseModel):
        items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
    else:
        assert isinstance(target, dict)
        items = target.items()
    for name, attr in items:
        if isinstance(attr, Swept):
            return True
        elif isinstance(attr, BaseModel | dict):
            if has_sweep(attr):
                return True
    return False
