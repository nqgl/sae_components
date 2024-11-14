import torch
import tqdm
from load import root_eval
from saeco.misc.nnsite import getsite, setsite
from torch import Tensor

e = root_eval.average_aggregated_patching_effect_on_dataset(605, batch_size=8)

root_eval.detokenize(e.topk(5).indices)
print()
