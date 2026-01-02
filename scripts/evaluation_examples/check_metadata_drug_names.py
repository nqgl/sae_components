# %%
from load_comlm_tahoe import root_eval


drugs = [
    "ralimetinib",
    "erlotinib",
    "gefitinib",
    "osimertinib",
    "PH-797804",
    # "MAPK",
]


drug_keys = root_eval.metadata_store.get("drug").info.fromstr.keys()
# %%
drug_keys
# %%
l = []
for d in drugs:
    e = []
    for dk in drug_keys:
        if d.lower() in dk.lower():
            e.append(dk)
    l.extend(e)
# %%
l
# %%
