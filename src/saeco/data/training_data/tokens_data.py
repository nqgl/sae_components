from functools import cached_property
from typing import TYPE_CHECKING, Any

import einops
import torch
import tqdm
from attrs import define

from saeco.data.config.split_config import SplitConfig
from saeco.data.config.tokenization_config import PackingMode, TokenizationMode
from saeco.data.piler import Piler
from saeco.data.piler.dict_piler import DictPiler
from saeco.data.training_data.on_the_fly_tokenizer import OnTheFlyTokenizer
from saeco.data.training_data.tokens_data_interface import TokensDataInterface

if TYPE_CHECKING:
    from saeco.data.config.data_cfg import DataConfig


@define
class TokensData(TokensDataInterface[torch.Tensor]):
    cfg: "DataConfig[Any ]"
    split: SplitConfig

    @cached_property
    def src_dataset_data(self):
        dataset = self.cfg.load_dataset_from_split(self.split)
        data = dataset[self.cfg.tokens_column_name]
        if not isinstance(data, torch.Tensor):
            data = data[:]
        assert isinstance(data, torch.Tensor)

        dataset_document_length = data.shape[1]
        if dataset_document_length % self.seq_len != 0:
            tqdm.tqdm.write(
                f"Document length {dataset_document_length} is not a multiple of the requested sequence length {self.seq_len}, truncating documents"
            )
            input("Press enter to continue and acknowledge this warning")
            data = data[:, : self.seq_len * (dataset_document_length // self.seq_len)]

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
        if self.dataset_document_length < self.seq_len:
            raise ValueError(
                f"Document length {self.dataset_document_length} is less than the requested sequence length {self.seq_len}"
            )
        # Only force first-token BOS for the pretokenized path. When tokenizing
        # on-the-fly, the tokenizer (or chat template) already injects BOS
        # where appropriate — overwriting here would produce double-BOS or
        # clobber the first real token of a packed sequence.
        if (
            self.cfg.set_bos
            and self.cfg.tokenization.mode == TokenizationMode.PRETOKENIZED
        ):
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
                compress=True,
            )
        return Piler.open(
            self.cfg._tokens_piles_path(self.split),
        )

    def _tokens_dict_piler(self, write: bool, num_tokens: int | None = None) -> DictPiler:
        path = self.cfg._tokens_piles_path(self.split)
        if write:
            assert num_tokens is not None
            return DictPiler.create(
                path,
                dtypes={"input_ids": torch.int64, "attention_mask": torch.bool},
                fixed_shapes={
                    "input_ids": [self.seq_len],
                    "attention_mask": [self.seq_len],
                },
                num_piles=(
                    1 + num_tokens // self.cfg.generation_config.tokens_per_pile
                ),
            )
        return DictPiler.open(path)

    def _store_split(self, split: SplitConfig):
        mode = self.cfg.tokenization.mode
        packing = self.cfg.tokenization.packing
        if mode == TokenizationMode.PRETOKENIZED:
            self._store_split_pretokenized(split)
        elif packing == PackingMode.PAD:
            self._store_split_on_the_fly_pad(split)
        else:
            self._store_split_on_the_fly_tensor(split)

    def _store_split_pretokenized(self, split: SplitConfig):
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

    def _on_the_fly_tokenizer(self) -> OnTheFlyTokenizer:
        return OnTheFlyTokenizer(
            cfg=self.cfg,
            split=self.split,
            tokenizer=self.cfg.model_cfg.model_load_cfg.tokenizer,
        )

    def _on_the_fly_yield_rows(self) -> int:
        # Tie yield batch size to the tokenization map batch size; this governs
        # how many rows are buffered in memory before a piler.distribute call.
        return max(1, self.cfg.tokenization.map_batch_size)

    def _store_split_on_the_fly_tensor(self, split: SplitConfig):
        tqdm.tqdm.write(
            f"Tokenizing on the fly ({self.cfg.tokenization.mode.value},"
            f" {self.cfg.tokenization.packing.value}) for {split.split}"
        )
        otf = self._on_the_fly_tokenizer()
        num_tokens_estimate = otf.estimate_num_tokens()
        piler = self.tokens_piler(write=True, num_tokens=num_tokens_estimate)
        for batch in otf.iter_tensor_batches(self._on_the_fly_yield_rows()):
            piler.distribute(batch)
        piler.shuffle_piles()

    def _store_split_on_the_fly_pad(self, split: SplitConfig):
        tqdm.tqdm.write(
            f"Tokenizing on the fly (pad) for {split.split}"
        )
        otf = self._on_the_fly_tokenizer()
        num_tokens_estimate = otf.estimate_num_tokens()
        dict_piler = self._tokens_dict_piler(write=True, num_tokens=num_tokens_estimate)
        for batch in otf.iter_dict_batches(self._on_the_fly_yield_rows()):
            dict_piler.distribute(batch)
        dict_piler.shuffle_piles()

    def get_tokens(self, num_tokens: int | None = None) -> torch.Tensor:
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
        assert isinstance(tokens, torch.Tensor)
        return (
            tokens[: num_tokens // tokens.shape[1] + 1]
            if num_tokens is not None
            else tokens
        )


@define
class PermutedDocs:
    cfg: "DataConfig"
    split: SplitConfig

    @cached_property
    def dataset(self):
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

    @cached_property
    def perm(self) -> torch.Tensor:
        return torch.randperm(self.dataset.num_rows)

    # def get_docs(self, num_docs=None):
    #     i = self.perm[:num_docs] if num_docs is not None else self.perm
    #     return self.dataset[i][self.cfg.tokens_column_name][:, : self.cfg.seq_len]

    # def get_docs_and_columns(self, num_docs=None, columns=[]):
    #     i = self.perm[:num_docs] if num_docs is not None else self.perm
    #     ds = self.dataset[i]
    #     return ds[self.cfg.tokens_column_name][:, : self.cfg.seq_len], {
    #         col: ds[col] for col in columns
    #     }

    def iter_docs_and_columns(self, batch_size, columns=[]):
        # cols =
        for i in range(0, len(self.perm) // batch_size * batch_size, batch_size):
            ds = self.dataset[self.perm[i : i + batch_size]]
            yield (
                ds[self.cfg.tokens_column_name][:, : self.cfg.seq_len],
                {col: ds[col] for col in columns},
            )
