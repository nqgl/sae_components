from pathlib import Path
from typing import Optional

import datasets
import torch
from pydantic import Field
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from saeco.data.acts_data import ActsData, ActsDataset
from saeco.data.bufferized_iter import bufferized_iter
from saeco.data.generation_config import DataGenerationProcessConfig

from saeco.data.locations import DATA_DIRS
from saeco.data.model_cfg import ModelConfig
from saeco.data.piler import Piler
from saeco.data.piler.dict_piler import DictBatch, DictPiler
from saeco.data.split_config import SplitConfig
from saeco.data.tokens_data import TokensData
from saeco.sweeps import SweepableConfig


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
    perm_all: bool = False
    databuffer_num_workers: int = 4

    # on the remote this wants to be ~32
    # on local that's more than necessary
    # maybe shouldn't be part of data config
    # since it doesn't affect training dynamics directly
    # and best values varies depending on hardware
    databuffer_queue_size: int | None = 32
    databuffer_worker_queue_base_size: int | None = 1
    databuffer_worker_offset_mult: int | None = 2
    databuffer_queue_size: int | None = 32
    databuffer_worker_queue_base_size: int | None = 1
    databuffer_worker_offset_mult: int | None = 2

    def idstr(self) -> str:
        seq_len = str(self.seq_len) if self.seq_len is not None else "null"
        fromdisk = "fromdisk_" if self.load_from_disk else ""
        extra_strs = fromdisk + seq_len
        extra_strs += "_perm" if self.perm_all else ""
        extra_strs += (
            f"_kwargs{self.model_cfg.model_kwargs}"
            if self.model_cfg.model_kwargs
            else ""
        )
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

    def _tokens_piles_path(self, split: SplitConfig) -> Path:
        return self._get_tokens_split_path(split) / "piles"

    def _tokens_perm_path(self, split: SplitConfig) -> Path:
        return self._get_tokens_split_path(split) / "perm.safetensors"

    def _acts_piles_path(self, split: SplitConfig) -> Path:
        return self._get_acts_split_path(split) / "piles"

    def acts_piler(
        self, split: SplitConfig, write=False, target_gb_per_pile=2, num_tokens=None
    ) -> DictPiler:
        if write:
            num_piles = self.generation_config.num_act_piles(num_tokens)
            sites = self.model_cfg.acts_cfg.sites
            dtypes = {site: self.model_cfg.acts_cfg.storage_dtype for site in sites}
            fixed_shapes = (
                {site: [self.model_cfg.acts_cfg.d_data] for site in sites}
                if not self.model_cfg.acts_cfg.site_d_datas
                else {
                    k: [v] for k, v in zip(sites, self.model_cfg.acts_cfg.site_d_datas)
                }
            )
            return DictPiler.create(
                self._acts_piles_path(split),
                dtypes=dtypes,
                fixed_shapes=fixed_shapes,
                num_piles=num_piles,
            )
        return DictPiler.open(self._acts_piles_path(split))

    def train_data_batch_generator(  # unused
        self,
        model,
        batch_size,
        nsteps=None,
        input_sites: list[str] | None = None,
        target_sites: list[str] | None = None,
    ):
        return ActsData(self, model).acts_generator(
            self.trainsplit,
            batch_size=batch_size,
            nsteps=nsteps,
            input_sites=input_sites,
            target_sites=target_sites,
        )

    def _train_dataset(
        self,
        model,
        batch_size,
        input_sites: list[str] | None = None,
        target_sites: list[str] | None = None,
    ):
        return ActsDataset(
            ActsData(self, model),
            self.trainsplit,
            batch_size,
            input_sites=input_sites,
            target_sites=target_sites,
        )

    def _get_queued_databuffer(
        self,
        batch_size,
        num_workers=None,
        queue_size=None,
        input_sites=None,
        target_sites=None,
    ):
        queue_size = queue_size or self.databuffer_queue_size
        num_workers = (
            self.databuffer_num_workers if num_workers is None else num_workers
        )
        buf = iter(
            self._get_databuffer(
                num_workers=num_workers,
                batch_size=batch_size,
                input_sites=input_sites,
                target_sites=target_sites,
            )
        )
        if queue_size is not None:
            return bufferized_iter(
                buf,
                queue_size=queue_size,
                getnext=lambda b: next(b).cuda(non_blocking=True),
            )
        return buf

    def _get_databuffer(
        self, num_workers=0, batch_size=4096, input_sites=None, target_sites=None
    ):
        model = None
        if not self._acts_piles_path(self.trainsplit).exists():
            model = self.model_cfg.model
        ds = self._train_dataset(
            model,
            batch_size=batch_size,
            input_sites=input_sites,
            target_sites=target_sites,
        )
        dl = DataLoader(
            ds, num_workers=num_workers, shuffle=False, pin_memory=True, batch_size=None
        )

        return dl

    def get_split_tokens(self, split, num_tokens=None):  ###
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

    def getsplit(self, split: str):
        return getattr(self, f"{split}split")

    def load_dataset_from_split(self, split: SplitConfig, to_torch=True):
        if self.perm_all:
            perm_path = (
                DATA_DIRS._CHUNKS_DIR
                / self.idstr()
                / self.model_cfg.modelstring
                / "perm.safetensors"
            )
            assert not self.load_from_disk
            dataset = datasets.load_dataset(
                self.dataset,
                split=split.split,
                cache_dir=DATA_DIRS.CACHE_DIR,
            )
            if not perm_path.exists():
                perm_path.parent.mkdir(parents=True, exist_ok=True)
                perm = torch.randperm(dataset.shape[0])
                save_file({"perm": perm}, perm_path)
            else:
                perm = load_file(perm_path)["perm"]
            start = int(len(dataset) * split.start / 100)
            end = int(len(dataset) * split.end / 100)
            split_perm = perm[start:end]
            dataset = dataset.select(split_perm)
        else:
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
            dataset.set_format(
                type="torch", columns=[self.tokens_column_name], output_all_columns=True
            )
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
