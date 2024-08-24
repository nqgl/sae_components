from pydantic import Field
from transformer_lens import HookedTransformer
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.split_config import SplitConfig
from saeco.data.model_cfg import ModelConfig
from saeco.data.tabletensor import Piler
import datasets
import torch

from saeco.sweeps import SweepableConfig
from typing import Optional

from saeco.data.locations import DATA_DIRS

import einops
import tqdm


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
        dl = torch.utils.data.DataLoader(ds, num_workers=num_workers)

        def squeezeyielder():
            for bn in dl:
                yield bn.cuda().squeeze(0)

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


class TokensData:
    def __init__(self, cfg: DataConfig, model: HookedTransformer, split: SplitConfig):
        self.cfg = cfg
        self.model = model
        self.split = split
        self._data = None
        self._documents = None

    @property
    def src_dataset_data(self):
        if self._data is not None:
            return self._data
        dataset = self.cfg.load_dataset_from_split(self.split)
        self._data = dataset[self.cfg.tokens_column_name]
        assert self.src_dataset_data.ndim == 2
        if self.dataset_document_length < self.seq_len:
            raise ValueError(
                f"Document length {self.dataset_document_length} is less than the requested sequence length {self.seq_len}"
            )
        if self.dataset_document_length % self.seq_len != 0:
            tqdm.tqdm.write(
                f"Document length {self.dataset_document_length} is not a multiple of the requested sequence length {self.seq_len}, truncating documents"
            )
            input("Press enter to continue and acknowledge this warning")
            self._data = self.src_dataset_data[
                :, : self.seq_len * (self.dataset_document_length // self.seq_len)
            ]
        return self._data

    @property
    def dataset_document_length(self):
        return self.src_dataset_data.shape[1]

    @property
    def seq_len(self):
        return self.cfg.seq_len or self.dataset_document_length

    @property
    def documents(self) -> torch.Tensor:
        if self._documents is not None:
            return self._documents
        if self.dataset_document_length != self.seq_len:
            self._documents = einops.rearrange(
                self.src_dataset_data,
                "batch (x seq_len) -> (batch x) seq_len",
                x=self.dataset_document_length // self.seq_len,
                seq_len=self.seq_len,
            )
        else:
            self._documents = self.src_dataset_data
        if self.cfg.set_bos:
            self._documents[:, 0] = self.model.tokenizer.bos_token_id
        return self._documents

    @property
    def num_tokens(self):
        return self.documents.numel()

    def tokens_piler(self, write=False, num_tokens=None) -> Piler:
        if write and num_tokens is None:
            raise ValueError("num_tokens must be specified if write=True")
        if num_tokens is not None and (not write):
            raise ValueError("num_tokens was specified but write=False")
        return Piler(
            self.cfg._tokens_piles_path(self.split),
            dtype=torch.int64,
            fixed_shape=[self.seq_len],
            num_piles=(
                1 + num_tokens // self.cfg.generation_config.tokens_per_pile
                if write
                else None
            ),
        )

    def _store_split(self, split: SplitConfig):
        tqdm.tqdm.write(f"Storing tokens for {split.split}")
        piler = self.tokens_piler(write=True, num_tokens=self.num_tokens)
        tqdm.tqdm.write("Distributing tokens to piles")
        doc_dist_batch_size = (
            self.documents.shape[0]
            // self.cfg.generation_config.num_document_distribution_batches
        )
        for i in tqdm.tqdm(
            range(
                0,
                self.documents.shape[0] // doc_dist_batch_size * doc_dist_batch_size,
                doc_dist_batch_size,
            )
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
        assert (
            num_piles <= piler.num_piles
        ), f"{num_tokens}, {self.cfg.generation_config.tokens_per_pile}, {piler.num_piles}"
        tokens = piler[0:num_piles]
        assert (
            num_tokens is None
            or abs(tokens.numel() - num_tokens)
            < self.cfg.generation_config.tokens_per_pile
        ), f"{tokens.shape} from piler vs {num_tokens} requested\
                this is expected if tokens per split is small, otherwise a bug.\
                    \n piles requested: {num_piles}, available: {piler.num_piles}"
        return (
            tokens[: num_tokens // tokens.shape[1] + 1]
            if num_tokens is not None
            else tokens
        )


class ActsData:
    def __init__(self, cfg: DataConfig, model: HookedTransformer):
        self.cfg = cfg
        self.model = model

    def _store_split(self, split: SplitConfig):
        tokens_data = TokensData(self.cfg, self.model, split=split)
        tokens = tokens_data.get_tokens(num_tokens=split.tokens_from_split)
        acts_piler = self.cfg.acts_piler(
            split,
            write=True,
            num_tokens=split.tokens_from_split or tokens_data.num_tokens,
        )

        tqdm.tqdm.write(f"Storing acts for {split.get_split_key()}")

        meta_batch_size = (
            self.cfg.generation_config.meta_batch_size // tokens_data.seq_len
        )
        tokens_split = tokens.split(meta_batch_size)
        for acts in tqdm.tqdm(
            self.acts_generator_from_tokens_generator(
                tokens_split,
                llm_batch_size=self.cfg.generation_config.llm_batch_size
                // tokens_data.seq_len,
            ),
            total=len(tokens_split),
        ):
            acts_piler.distribute(acts)
        acts_piler.shuffle_piles()

    def acts_generator_from_tokens_generator(self, tokens_generator, llm_batch_size):
        for tokens in tokens_generator:
            acts = self.to_acts(tokens, llm_batch_size=llm_batch_size)
            yield acts

    def to_acts(self, tokens, llm_batch_size, rearrange=True, skip_exclude=False):
        acts_list = []
        # assert tokens.shape[0] % llm_batch_size == 0
        with torch.autocast(device_type="cuda"):
            with torch.inference_mode():

                def hook_fn(acts, hook):
                    acts_list.append(acts)

                for i in range(
                    0,
                    tokens.shape[0],
                    llm_batch_size,
                ):
                    self.model.run_with_hooks(
                        tokens[i : i + llm_batch_size],
                        **self.cfg.model_cfg.model_kwargs,
                        stop_at_layer=self.cfg.model_cfg.acts_cfg.layer_num + 1,
                        fwd_hooks=[(self.cfg.model_cfg.acts_cfg.hook_site, hook_fn)],
                    )
        acts = torch.cat(acts_list, dim=0).half()
        if self.cfg.model_cfg.acts_cfg.excl_first and not skip_exclude:
            acts = acts[:, 1:]
        if rearrange:
            acts = einops.rearrange(
                acts,
                "batch seq d_data -> (batch seq) d_data",
            )
        return acts

    def acts_generator(
        self, split: SplitConfig, batch_size, nsteps=None, id=None, nw=None
    ):
        if not self.cfg._acts_piles_path(split).exists():
            self._store_split(split)
        assert id == nw == None or id is not None and nw is not None
        id = id or 0
        nw = nw or 1
        piler = self.cfg.acts_piler(split)
        assert (
            nsteps is None
            or split.tokens_from_split is None
            or nsteps <= split.tokens_from_split // batch_size
        )
        print("\nProgress bar activation batch count is approximate\n")
        progress = (
            tqdm.trange(nsteps or split.tokens_from_split // batch_size)
            if split.tokens_from_split is not None
            else None
        )
        for p in range(id, piler.num_piles, nw):
            print("get next pile")
            print(id, nw, p)
            pile = piler[p]
            if progress is None:
                progress = tqdm.trange(pile.shape[0] * piler.num_piles // batch_size)

            assert pile.dtype == torch.float16
            print("got next pile")
            for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                yield pile[i : i + batch_size]
                progress.update()
            if nsteps is not None and progress.n >= nsteps:
                break


class ActsDataset(torch.utils.data.IterableDataset):
    def __init__(self, acts: ActsData, split: SplitConfig, batch_size):
        self.acts = acts
        self.split = split
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return self.acts.acts_generator(self.split, self.batch_size)

        return self.acts.acts_generator(
            self.split, self.batch_size, id=worker_info.id, nw=worker_info.num_workers
        )
