
import torch

# def max(t:torch.Tensor, dim:int):


def sparse_to_mask(t):
    v = torch.ones_like(t.values(), dtype=torch.bool)
    return torch.sparse_coo_tensor(t.indices(), v, t.shape)


def indices_to_sparse_mask(indices, shape):
    v = torch.ones(indices.shape[1], dtype=torch.bool)
    for p in range(indices.shape[0], len(shape)):
        v = v.unsqueeze(-1).expand(-1, shape[p])
    return torch.sparse_coo_tensor(indices, v, shape)


def overlap_mask(t1, t2):
    m1 = sparse_to_mask(t1)
    m2 = sparse_to_mask(t2)
    return m1 * m2
