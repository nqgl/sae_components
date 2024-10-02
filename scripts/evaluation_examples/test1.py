import torch
from load import root_eval

feat = 43
docs, acts, metas, doc_ids = root_eval.top_activations_and_metadatas(
    feat,
    k=25,
    metadata_keys=["third"],
    return_str_docs=True,
)
acts = acts.to_dense()
assert root_eval.docstrs[doc_ids] == docs
# for i in range(25):
#     acts2 = root_eval.acts[doc_ids[i]].to_dense()[:, :, feat]
#     print((acts2 == acts[i].to_sparse_coo()).all())
sdi = doc_ids.argsort()
acts2 = root_eval.acts[doc_ids].to_dense()[:, :, feat]
acts3 = root_eval.acts[doc_ids[sdi]].to_dense()[:, :, feat]
(acts2 == acts3).all()
assert (acts[sdi] == acts2).all()
assert (acts == root_eval.features[feat].to_dense()[doc_ids.unsqueeze(0)]).all()
assert (root_eval.metadatas["third"][doc_ids] == metas[0]).all()
assert (
    acts.flatten().topk(25).values
    == root_eval.features[feat]
    .to_dense()
    .value.flatten()
    .topk(
        25,
    )
    .values
).all()


fe = root_eval.open_filtered("test_filter")

docs, acts, metas, doc_ids = fe.top_activations_and_metadatas(
    feat,
    k=25,
    metadata_keys=["third"],
    return_str_docs=True,
)
acts = acts.to_dense()
assert root_eval.docstrs[doc_ids] == docs
# for i in range(25):
#     acts2 = root_eval.acts[doc_ids[i]].to_dense()[:, :, feat]
#     print((acts2 == acts[i]).all())
sdi = doc_ids.argsort()
acts2 = root_eval.acts[doc_ids].to_dense()[:, :, feat]
acts3 = root_eval.acts[doc_ids[sdi]].to_dense()[:, :, feat]
(acts2 == acts3).all()
assert (acts[sdi] == acts2).all()
assert (acts == root_eval.features[feat].to_dense()[doc_ids.unsqueeze(0)]).all()
assert (root_eval.metadatas["third"][doc_ids] == metas[0]).all()
assert (
    acts.flatten().topk(25).values
    == fe.features[feat]
    .to_dense()
    .value.flatten()
    .topk(
        25,
    )
    .values
).all()


from nqgl.mlutils.profiling.time_gpu import TimedFunc

TimedFunc(root_eval.count_token_occurrence1)()
TimedFunc(root_eval.count_token_occurrence1)()
TimedFunc(root_eval.count_token_occurrence)()
TimedFunc(root_eval.count_token_occurrence)()
TimedFunc(root_eval.count_token_occurrence3)()
TimedFunc(root_eval.count_token_occurrence3)()
TimedFunc(root_eval.count_token_occurrence)()
TimedFunc(root_eval.count_token_occurrence1)()
TimedFunc(root_eval.count_token_occurrence3)()
