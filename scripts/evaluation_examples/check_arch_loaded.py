# %%
from functools import cached_property
from typing import get_origin

from saeco.data.config._comlm_data_config_definitions import saeco_tahoe_data_cfg

from saeco.architecture.arch_prop import FieldsLoaded

# "trainable" in root_eval.architecture.__dict__


# %%
print()
from load_comlm_tahoe import root_eval

# assert not FieldsLoaded.check_fields_loaded(root_eval.architecture).loaded

# class FieldsLoaded(BaseModel):
#     loaded: list[str]
#     not_loaded: list[str]

#     @classmethod
#     def check_fields_loaded(cls, instance):
#         if type(instance) not in _fields_dict:
#             return cls(loaded=[], not_loaded=[])
#         field_names = [
#             field
#             for fields_l in _fields_dict[type(instance)].values()
#             for field in fields_l
#         ]
#         loaded = [field for field in field_names if field in instance.__dict__]
#         not_loaded = [field for field in field_names if field not in instance.__dict__]
#         assert set(loaded) | set(not_loaded) == set(field_names)
#         assert set(loaded) & set(not_loaded) == set()

#         return cls(
#             loaded=loaded,
#             not_loaded=not_loaded,
#         )

# # %%
# loaded = FieldsLoaded.check_fields_loaded(root_eval.architecture)
# %%
# loaded.loaded
# # %%
# loaded.not_loaded
# # %%
# root_eval.sae
# # %%
# at = type(root_eval.architecture)

# get_type_hints(at)
# %%
# from functools import cached_property

# for k, v in at.__dict__.items():
#     if isinstance(v, cached_property):
#         print(k)
# # %%
# at.__dict__["_core_model"]
# # %%
# root_eval.architecture.__dict__["_core_model"]
# # %%
# at._core_model
# # %%
# at.__bases__[0].__dict__["_core_model"]


# %%
def get_cached_properties(cls: type) -> set[str]:
    if not isinstance(cls, type):
        cls = get_origin(cls)
    return {k for k in dir(cls) if isinstance(getattr(cls, k), cached_property)}


def loaded_cached_properties(inst):
    return {k for k in get_cached_properties(type(inst)) if k in inst.__dict__}


def check_cached_properties(inst):
    loaded = loaded_cached_properties(inst)
    not_loaded = get_cached_properties(type(inst)) - loaded
    return loaded, not_loaded


check_cached_properties(root_eval)
get_cached_properties(type(root_eval))
root_eval.sae_cfg.train_cfg.data_cfg = saeco_tahoe_data_cfg  # type: ignore
# %%
FieldsLoaded.check_fields_loaded(root_eval)
# %%
FieldsLoaded.check_fields_loaded(root_eval.architecture)
# %%
check_cached_properties(root_eval.architecture)
# %%
