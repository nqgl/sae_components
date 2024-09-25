from pathlib import Path

import torch
from attr import define, field
from torch import Tensor


@define
class NamedFilter:
    filter: Tensor
    filter_name: str
    # cfg: CachingConfig

    def chunk_filters(self, cfg):
        chunked = self.filter.chunk(cfg.num_chunks)
        assert len(chunked) == cfg.num_chunks
        return chunked

    def filter_chunk_output(self, output, chunk):
        return output[self.chunk_filters(chunk.cfg)[chunk.idx]]

    def save_filter(self, root_path):
        torch.save(self.filter, root_path / "filtered" / f"{self.filter_name}.pt")

    # def mask(self, tensor, chunk=None):
    #     if chunk is None:
    #         mask = self.filter
    #     else:
    #         mask = self.chunk_filters(chunk.cfg)[chunk.idx]
    #     return FilteredTensor.from_unmasked_value(tensor, mask)
