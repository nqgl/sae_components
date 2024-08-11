import saeco.components as co
from saeco.components.ops.fnlambda import Lambda


import torch


def TopK(k):
    k = int(k)

    def _topk(x):
        v, i = x.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(x).scatter_(-1, i, v)

    return Lambda(_topk)


def TopKDead(k, freq_tracker, threshold):
    k = int(k)

    def _topkdead(x):
        mask = freq_tracker.freqs < threshold
        x = torch.where(mask.unsqueeze(0), x, torch.zeros_like(x))
        if mask.sum() <= k:
            return x
        v, i = x.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(x).scatter_(-1, i, v)

    return Lambda(_topkdead)


class NormalizedResidL2Loss(co.Loss):
    def loss(self, x, y, y_pred, cache: co.SAECache):
        loss = (y - y_pred).pow(2).mean() / cache._ancestor.L2_loss.detach()
        if loss.isnan().any() or loss.isinf().any():
            return torch.zeros(1)
        return loss
