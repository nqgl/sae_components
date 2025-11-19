
from load import root_eval

# null_metadata_names = ["tissue", "cell_type"]

# for t in null_metadata_names:
#     # b = torch.zeros(root_eval.num_docs, dtype=torch.long)
#     # root_eval.metadatas[t] = b
#     root_eval.metadatas.set_str_translator(t, {"0": 0})

# root_eval._get_feature_families_unlabeled._version
all_families = root_eval.get_feature_families()

len(all_families.levels[0].families)
sum([len(f.subfamilies) for f in all_families.levels[0].families.values()])
all_families.levels[0].families[44].subfamilies
levels, families = root_eval.generate_feature_families2(doc_agg="count", threshold=0.01)
(levels[1] == levels[2]).all()

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
all_families
