import sys

import torch
from load import root_eval


levels, families = root_eval.generate_feature_families2()
(levels[1] == levels[2]).all()
all_families = root_eval.get_feature_families()
fams2 = [all_families.levels[0].families[0], all_families.levels[0].families[1]]
res = root_eval.top_overlapped_feature_family_documents(
    families=fams2, k=100, return_str_docs=True
)
filt = root_eval.open_filtered("test_filter")
all_families = filt.get_feature_families()
fams2 = [all_families.levels[0].families[0], all_families.levels[0].families[1]]
res = filt.top_overlapped_feature_family_documents(
    families=fams2, k=100, return_str_docs=True
)
print()
for i in range(40):
    print(z[i].count_nonzero())
