import nqgl.mlutils.profiling.time_gpu as tg
import torch
from load import root_eval

# ec.filter_docs()


def time_indexing(index_fn):
    getnum = 20000
    doc_ids = torch.arange(getnum)
    doc_ids[getnum // 2 :] += 123
    doc_ids = doc_ids[tg.TimedFunc(torch.randperm)(getnum)]
    r = tg.TimedFunc(index_fn)(doc_ids)

    return r, doc_ids


def checker(a1, a2):
    getnum = 1000
    r, doc_ids = a1
    r2, doc_ids2 = a2
    for i in range(0, getnum, getnum // 10):
        assert (
            r2[i] - r[torch.arange(getnum)[doc_ids == doc_ids2[i]].item()]
        ).abs().sum() == 0, i

        # assert (r[i] - r2[doc_ids[doc_ids[i] == doc_ids].item()]).abs().sum() == 0, i


# a1 = time_indexing(ec.saved_acts.acts.document_select_sparse2)


def index_tokens_by_bool_conversion(indices):
    mask = torch.zeros(root_eval.cache_cfg.num_docs, dtype=torch.bool)
    mask[indices] = True
    return root_eval.filter_docs(mask, only_return_selected=True)


def index_by_bool_conversion(indices):
    mask = torch.zeros(root_eval.cache_cfg.num_docs, dtype=torch.bool)
    mask[indices] = True
    return root_eval.filter_acts(mask, only_return_selected=True)


# a2 = time_indexing(index_by_bool_conversion)

# checker(a1, a2)

# ec.saved_acts.acts[document][position]
# ec.saved_acts.tokens[0:5]
# patching
# grad patching


@tg.timedfunc_wrapper()
def index_by_feature_docs(i):
    f = root_eval.get_feature(i)
    mask = f.cuda().to_dense().value > 0
    mask = mask.any(dim=1)
    assert mask.ndim == 1
    return root_eval.filter_docs(mask, only_return_selected=False)


@tg.timedfunc_wrapper()
def index_by_feature(i):
    f = root_eval.get_feature(i)
    mask = f.cuda().to_dense().value > 0
    return root_eval.filter_docs(mask, only_return_selected=False, seq_level=True)


@tg.timedfunc_wrapper()
def index_acts_by_feature_docs(i):
    f = root_eval.get_feature(i)
    mask = f.cuda().to_dense().value > 0
    mask = mask.any(dim=1)
    assert mask.ndim == 1
    return root_eval.filter_acts(mask, only_return_selected=False)


f = index_by_feature_docs(5)
f2 = index_by_feature(5)
af = index_acts_by_feature_docs(5)
print()


root_eval.forward_ad_with_sae()
