from load import ec
import torch
import nqgl.mlutils.profiling.time_gpu as tg


doc_ids = torch.arange(10)
doc_ids[5:] += 200
doc_ids = doc_ids[torch.randperm(10)]
r = ec.saved_acts.tokens.document_select(doc_ids)
print(r.shape)
print(doc_ids)

print()

doc_ids2 = torch.arange(10)
doc_ids2[5:] += 200
doc_ids2 = doc_ids2[torch.randperm(10)]
r2 = ec.saved_acts.tokens.document_select(doc_ids2)
print(r.shape)
# print(r.indices())
for i in range(10):
    assert (
        r2[i] - r[torch.arange(10)[doc_ids == doc_ids2[i]].item()]
    ).abs().sum() == 0, i
r[1] - r2[0]
[(r[0] - r2[i]).sum().item() for i in range(10)]
doc_ids2
doc_ids
########
# print(r.shape)
# print(doc_ids)

# print()

getnum = 1000
doc_ids = torch.arange(getnum)
doc_ids[getnum // 2 :] += 123
doc_ids2 = torch.arange(getnum)
doc_ids2[getnum // 2 :] += 123
doc_ids2 = doc_ids2[torch.randperm(getnum)]
r2 = tg.TimedFunc(ec.saved_acts.acts.document_select_sparse)(doc_ids2)
r = tg.TimedFunc(ec.saved_acts.acts.document_select_sparse_sorted)(doc_ids)
# print(r.shape)
# print(r.indices())
for i in range(getnum):
    assert (
        r2[i] - r[torch.arange(getnum)[doc_ids == doc_ids2[i]].item()]
    ).abs().sum() == 0, i
r[1] - r2[0]
[(r[2] - r2[i]).sum().item() for i in range(10)]
doc_ids2
doc_ids

getnum = 2000
mul = 1
doc_ids = torch.arange(0, getnum * mul, mul)
doc_ids[getnum // 2 :] += 123
doc_ids2 = torch.arange(0, getnum * mul, mul)
doc_ids2[getnum // 2 :] += 123
doc_ids2 = doc_ids2[torch.randperm(getnum)]
r2 = tg.TimedFunc(ec.saved_acts.acts.document_select_sparse)(doc_ids2)
r = tg.TimedFunc(ec.saved_acts.acts.document_select_sparse_sorted)(doc_ids)
# print(r.shape)
# print(r.indices())
for i in range(0, getnum, 500):
    assert (
        r2[i] - r[torch.arange(getnum)[doc_ids == doc_ids2[i]].item()]
    ).abs().sum() == 0, i
r[1] - r2[0]
# [(r[2] - r2[i]).sum().item() for i in range(10)]
doc_ids2
doc_ids
