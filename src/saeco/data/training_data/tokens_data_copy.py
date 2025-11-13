from typing import TYPE_CHECKING

import einops
import torch
import tqdm
from nnsight import LanguageModel, NNsight
from torch import Tensor

from saeco.data.config.split_config import SplitConfig
from saeco.data.piler import Piler
from functools import cached_property
import datasets
from attrs import define, field


if TYPE_CHECKING:
    from saeco.data.config.data_cfg import DataConfig


@define
class TokensData:
    cfg: "DataConfig"
    split: SplitConfig

    @cached_property
    def src_dataset_data(self):
        dataset = self.cfg.load_dataset_from_split(self.split)
        data = dataset[self.cfg.tokens_column_name]
        assert data.ndim == 2
        if self.dataset_document_length < self.seq_len:
            raise ValueError(
                f"Document length {self.dataset_document_length} is less than the requested sequence length {self.seq_len}"
            )
        if self.dataset_document_length % self.seq_len != 0:
            tqdm.tqdm.write(
                f"Document length {self.dataset_document_length} is not a multiple of the requested sequence length {self.seq_len}, truncating documents"
            )
            input("Press enter to continue and acknowledge this warning")
            data = data[
                :, : self.seq_len * (self.dataset_document_length // self.seq_len)
            ]
        return data

    @property
    def dataset_document_length(self):
        return self.src_dataset_data.shape[1]

    @property
    def seq_len(self):
        return self.cfg.seq_len or self.dataset_document_length

    @cached_property
    def documents(self) -> torch.Tensor:
        if self.dataset_document_length != self.seq_len:
            docs = einops.rearrange(
                self.src_dataset_data,
                "batch (x seq_len) -> (batch x) seq_len",
                x=self.dataset_document_length // self.seq_len,
                seq_len=self.seq_len,
            )
        else:
            docs = self.src_dataset_data
        if self.cfg.set_bos:
            docs[:, 0] = self.cfg.model_cfg.tokenizer.bos_token_id
        return docs

    @property
    def num_tokens(self):
        return self.documents.numel()

    def tokens_piler(self, write=False, num_tokens=None) -> Piler:
        if write and num_tokens is None:
            raise ValueError("num_tokens must be specified if write=True")
        if num_tokens is not None and (not write):
            raise ValueError("num_tokens was specified but write=False")
        if write:
            return Piler.create(
                self.cfg._tokens_piles_path(self.split),
                dtype=torch.int64,
                fixed_shape=[self.seq_len],
                num_piles=(
                    1 + num_tokens // self.cfg.generation_config.tokens_per_pile
                ),
            )
        return Piler.open(
            self.cfg._tokens_piles_path(self.split),
        )

    def _store_split(self, split: SplitConfig):
        tqdm.tqdm.write(f"Storing tokens for {split.split}")
        piler = self.tokens_piler(write=True, num_tokens=self.num_tokens)
        tqdm.tqdm.write("Distributing tokens to piles")
        doc_dist_batch_size = (
            self.documents.shape[0]
            // self.cfg.generation_config.num_document_distribution_batches
        )
        for i in tqdm.trange(
            0,
            self.documents.shape[0] // doc_dist_batch_size * doc_dist_batch_size,
            doc_dist_batch_size,
        ):
            piler.distribute(self.documents[i : i + doc_dist_batch_size])
        piler.shuffle_piles()

    def get_tokens(self, num_tokens=None):
        if not self.cfg._tokens_piles_path(self.split).exists():
            self._store_split(self.split)
        piler = self.tokens_piler()

        num_piles = (
            piler.num_piles
            if num_tokens is None
            else (num_tokens + self.cfg.generation_config.tokens_per_pile - 1)
            // self.cfg.generation_config.tokens_per_pile
        )
        assert num_piles <= piler.metadata.num_piles, (
            f"{num_tokens}, {self.cfg.generation_config.tokens_per_pile}, {piler.num_piles}"
        )
        tokens = piler[0:num_piles]
        assert (
            num_tokens is None
            or abs(tokens.numel() - num_tokens)
            < self.cfg.generation_config.tokens_per_pile
        ), (
            f"{tokens.shape} from piler vs {num_tokens} requested\
                this is expected if tokens per split is small, otherwise a bug.\
                    \n piles requested: {num_piles}, available: {piler.num_piles}"
        )
        return (
            tokens[: num_tokens // tokens.shape[1] + 1]
            if num_tokens is not None
            else tokens
        )

    def get_tokens_iter(
        self,
        batch_size,
        id=None,
        nw=None,
    ):
        assert id == nw == None or id is not None and nw is not None
        id = id or 0
        nw = nw or 1
        if not self.cfg._tokens_piles_path(self.split).exists():
            self._store_split(self.split)

        piler = self.tokens_piler()
        for p in range(id % nw, piler.num_piles, nw):
            print("get next tokens pile")
            print(id, nw, p)
            pile = piler[p]
            for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                yield pile[i : i + batch_size]


@define
class PermutedDocs:
    cfg: "DataConfig"
    split: SplitConfig
    dataset: datasets.Dataset | datasets.DatasetDict = field(init=False)
    perm: Tensor = field(init=False)

    @dataset.default
    def _dataset_default_value(self):
        dataset = self.cfg.load_dataset_from_split(self.split)
        seq_len = dataset[0][self.cfg.tokens_column_name].shape[0]
        if self.cfg.seq_len:
            if seq_len < self.cfg.seq_len:
                raise ValueError(
                    f"Document length {seq_len} is less than the requested sequence length {self.cfg.seq_len}"
                )
            elif seq_len != self.cfg.seq_len:
                input(
                    f"Warning: document lengths {seq_len} is longer than seq_len, so documents will be truncated. Press enter to continue"
                )
        return dataset

    @perm.default
    def _perm_default_value(self):
        return torch.randperm(self.dataset.num_rows)

    @property
    def dataset_document_length(self):
        return self.src_dataset_data.shape[1]

    def get_docs(self, num_docs=None):
        i = self.perm[:num_docs] if num_docs is not None else self.perm
        return self.dataset[i][self.cfg.tokens_column_name][:, : self.cfg.seq_len]

    def get_docs_and_columns(self, num_docs=None, columns=[]):
        i = self.perm[:num_docs] if num_docs is not None else self.perm
        ds = self.dataset[i]
        return ds[self.cfg.tokens_column_name][:, : self.cfg.seq_len], {
            col: ds[col] for col in columns
        }

    def iter_docs_and_columns(self, batch_size, columns=[]):
        # cols =
        for i in range(0, len(self.perm) // batch_size * batch_size, batch_size):
            ds = self.dataset[self.perm[i : i + batch_size]]
            yield (
                ds[self.cfg.tokens_column_name][:, : self.cfg.seq_len],
                {col: ds[col] for col in columns},
            )
