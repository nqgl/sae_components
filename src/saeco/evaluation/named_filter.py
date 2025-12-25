import torch
from attrs import define, field
from torch import Tensor


@define
class NamedFilter:
    filter: Tensor = field()
    filter_name: str = field()

    @filter.validator
    def _validate_filter(self, attribute, value):
        assert isinstance(value, Tensor)
        assert value.dtype == torch.bool

    def chunk_filters(self, cfg):
        chunked = self.filter.chunk(cfg.num_chunks)
        assert len(chunked) == cfg.num_chunks
        return chunked

    def filter_chunk_output(self, output, chunk):
        return output[self.chunk_filters(chunk.cfg)[chunk.idx]]

    def filtered_dir(self, root_path):
        if self.filter_name is None:
            raise ValueError(
                "Current filter is unnamed. To use persistent storage on filtered evaluations, use a named, saved filter."
            )
        path = self._filtered_dir_from_root_and_name(root_path, self.filter_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _filtered_dir_from_root_and_name(cls, root_path, filter_name):
        return root_path / "filtered" / filter_name
