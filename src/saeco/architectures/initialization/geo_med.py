import torch
from saeco.core import Cache


def geometric_median(x, eps=1e-5, max_iter=100):
    """
    x: (batch, n, d)
    """
    y = torch.zeros_like(x[:, 0])
    print(y.shape)
    for i in range(max_iter):
        print(i)
        y0 = y
        y = (x / (y.unsqueeze(1) + eps)).sum(dim=1) / (
            ((y.unsqueeze(1) + eps)).sum(dim=1)
        )
        if torch.allclose(y0, y, atol=1e-8):
            break
    return y.squeeze(0)


def getmed(buf, normalizer, num_batches=10):
    big_batch = torch.cat(
        [
            normalizer.normalize(next(buf), cache=Cache())[0].float()
            for i in range(num_batches)
        ],
        dim=0,
    )
    med = geometric_median(big_batch.unsqueeze(0))
    assert not med.isnan().any()
    assert not med.isinf().any()
    assert med.ndim == 1
    return med


def getmean(buf, normalizer, num_batches=10):
    big_batch = torch.cat(
        [
            normalizer.normalize(next(buf), cache=Cache())[0].float()
            for i in range(num_batches)
        ],
        dim=0,
    )
    med = torch.mean(big_batch, dim=0)
    assert not med.isnan().any()
    assert not med.isinf().any()
    assert med.ndim == 1
    return med
