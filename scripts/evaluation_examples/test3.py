import sys

import torch
from load import root_eval


all_families = root_eval.cached_call.get_feature_families()
fams2 = [all_families.levels[0].families[0], all_families.levels[0].families[1]]
res = root_eval.top_overlapped_feature_family_documents(
    families=fams2, k=100, return_str_docs=True
)
print()
