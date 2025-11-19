from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from attrs import define

from saeco.data.config.split_config import SplitConfig
from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.data.piler.dict_piler import DictPiler
from saeco.data.training_data.tokens_data_interface import TokensDataInterface

if TYPE_CHECKING:
    from saeco.data.config.data_cfg import DataConfig


@define
class DictPiledTokensData(TokensDataInterface[DictBatch]):
    cfg: "DataConfig[Any]"
    piler_path: Path
    split: SplitConfig

    @property
    def seq_len(self) -> int:
        assert self.cfg.seq_len is not None
        return self.cfg.seq_len

    @property
    def num_tokens(self) -> int:
        assert self.cfg.seq_len is not None
        return self.cfg.seq_len * self.piler.num_samples

    @cached_property
    def piler(self) -> DictPiler:
        return DictPiler.open(self.piler_path)

    def get_tokens(self, num_tokens: int | None = None) -> DictBatch:
        start, end = self.split.get_bounds(self.piler.num_samples)
        num_samples = (
            num_tokens // self.seq_len if num_tokens is not None else end - start
        )
        if num_tokens is not None:
            num_available = self.seq_len * (end - start)
            if num_tokens > num_available:
                raise ValueError(
                    f"Requested {num_tokens} tokens, "
                    f"but only {num_available} are available"
                )
            end = start + num_samples

        return self.piler.sample_indexer[start:end]  # TODO: handle seq length
