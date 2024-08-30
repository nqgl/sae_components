from pydantic import Field
from saeco.data.acts_data import ActsData, ActsDataset
from saeco.data.tokens_data import TokensData
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.split_config import SplitConfig
from saeco.data.model_cfg import ModelConfig
from saeco.data.tabletensor import Piler
import datasets
import torch
from torch.utils.data import DataLoader
from saeco.sweeps import SweepableConfig
from typing import Optional

from saeco.data.locations import DATA_DIRS


# @dataclass
class DataConfig(SweepableConfig):
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    load_from_disk: bool = False
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    trainsplit: SplitConfig = Field(
        default_factory=lambda: SplitConfig(
            start=0,
            end=40,
            tokens_from_split=400_000_000,
        )
    )
    testsplit: SplitConfig = Field(
        default_factory=lambda: SplitConfig(
            start=80,
            end=90,
        )
    )
    valsplit: SplitConfig = Field(
        default_factory=lambda: SplitConfig(
            start=90,
            end=100,
        )
    )
    set_bos: bool = True
    seq_len: int | None = 128
    tokens_column_name: str = "input_ids"
    generation_config: DataGenerationProcessConfig = Field(
        default_factory=DataGenerationProcessConfig
    )

    def idstr(self):
        seq_len = str(self.seq_len) if self.seq_len is not None else "null"
        fromdisk = "fromdisk_" if self.load_from_disk else ""
        extra_strs = fromdisk + seq_len
        return f"{self.dataset.replace('/', '_')}_{extra_strs}_{self.set_bos}"

    def _get_tokens_split_path(self, split: SplitConfig):
        return (
            DATA_DIRS._CHUNKS_DIR
            / self.idstr()
            / self.model_cfg.modelstring
            / split.split_dir_id
            / "tokens"
        )

    def _get_acts_split_path(self, split: SplitConfig):
        return (
            DATA_DIRS._CHUNKS_DIR
            / self.idstr()
            / self.model_cfg.modelstring
            / split.split_dir_id
            / "acts"
        )

    def _tokens_piles_path(self, split: SplitConfig):
        return self._get_tokens_split_path(split) / "piles"

    def _acts_piles_path(self, split: SplitConfig):
        return self._get_acts_split_path(split) / "piles"

    def acts_piler(
        self, split: SplitConfig, write=False, target_gb_per_pile=2, num_tokens=None
    ) -> Piler:
        num_piles = None
        if write:
            num_piles = self.generation_config.num_act_piles(num_tokens)
        return Piler(
            self._acts_piles_path(split),
            dtype=torch.float16,
            fixed_shape=[self.model_cfg.acts_cfg.d_data],
            num_piles=(num_piles if write else None),
        )

        # loading_data_first_time = not dataset_reshaped_path.exists()

    def train_data_batch_generator(self, model, batch_size, nsteps=None):
        return ActsData(self, model).acts_generator(
            self.trainsplit, batch_size=batch_size, nsteps=nsteps
        )

    def train_dataset(self, model, batch_size):
        return ActsDataset(ActsData(self, model), self.trainsplit, batch_size)

    def get_databuffer(self, num_workers=0, batch_size=4096):
        ds = self.train_dataset(self.model_cfg.model, batch_size=batch_size)
        dl = DataLoader(
            ds,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

        def squeezeyielder():
            for bn in dl:
                yield bn.squeeze(0)

        return squeezeyielder()

    def get_split_tokens(self, split, num_tokens=None):
        return TokensData(
            self,
            self.model_cfg.model,
            split=(
                split
                if isinstance(split, SplitConfig)
                else getattr(self, f"{split}split")
            ),
        ).get_tokens(
            num_tokens=num_tokens,
        )

    def load_dataset_from_split(self, split: SplitConfig, to_torch=True):
        if self.load_from_disk:
            dataset = datasets.load_from_disk(
                self.dataset,
            )
        else:
            dataset = datasets.load_dataset(
                self.dataset,
                split=split.get_split_key(),
                cache_dir=DATA_DIRS.CACHE_DIR,
            )

        if to_torch:
            dataset.set_format(type="torch", columns=[self.tokens_column_name])
        return dataset

    def acts_data(self) -> "ActsData":
        return ActsData(self, self.model_cfg.model)

    def tokens_data(self, split="train") -> "TokensData":
        return TokensData(
            self,
            self.model_cfg.model,
            split=(
                split
                if isinstance(split, SplitConfig)
                else getattr(self, f"{split}split")
            ),
        )
