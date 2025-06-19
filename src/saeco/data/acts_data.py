from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import einops
import torch
import torch.utils
import torch.utils.data
import tqdm
from attrs import define
from nnsight import LanguageModel

from saeco.data.bufferized_iter import bufferized_iter
from saeco.data.piler.dict_piler import DictBatch
from saeco.data.split_config import SplitConfig
from saeco.data.tokens_data import TokensData
from saeco.misc import str_to_dtype
from saeco.misc.nnsite import getsite

if TYPE_CHECKING:
    from saeco.data.data_cfg import DataConfig


from saeco.data.sae_train_batch import SAETrainBatch


class ActsData:
    """
    Generates, stores, and loads activations
    """

    def __init__(self, cfg: "DataConfig", model: LanguageModel | None):
        self.cfg = cfg
        self.model = model

    def _store_chunk(self, chunk_idx: int, chunk_path: Path, split: SplitConfig):
        print(f"storing chunk {chunk_idx} at {chunk_path}")
        tokens_data = TokensData(self.cfg, self.model, split=split)
        tokens = tokens_data.get_tokens(
            num_tokens=split.acts_per_chunk * split.act_chunks_cached,
        )

        tokens_start = chunk_idx * split.acts_per_chunk
        tokens_end = (chunk_idx + 1) * split.acts_per_chunk
        tokens = tokens[tokens_start // tokens.shape[1] : tokens_end // tokens.shape[1]]

        tok_start_idx = tokens_start // tokens.shape[1]
        tok_idx = tokens_end // tokens.shape[1]

        acts_piler = self.cfg.acts_piler(
            chunk_path,
            split,
            write=True,
            num_tokens=split.acts_per_chunk or tokens_data.num_tokens,
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
            position=0,
            desc="Storing",
        ):
            acts_piler.distribute(acts.data)
        acts_piler.shuffle_piles()

    def acts_generator_from_tokens_generator(self, tokens_generator, llm_batch_size):
        for tokens in tokens_generator:
            acts = self.to_acts(tokens, llm_batch_size=llm_batch_size)
            yield acts

    @contextmanager
    def _null_context(self):
        yield

    def autocast_context(self):
        if self.cfg.model_cfg.acts_cfg.autocast_dtype is False:
            return self._null_context()
        return torch.autocast(
            device_type="cuda",
            dtype=(
                self.cfg.model_cfg.acts_cfg.autocast_dtype
                or self.cfg.model_cfg.torch_dtype
            ),
        )

    def to_acts(
        self,
        tokens,
        llm_batch_size,
        rearrange=True,
        skip_exclude=False,
        force_not_skip_padding=False,
        batched_kwargs={},
    ) -> DictBatch:
        assert isinstance(tokens, torch.Tensor)
        assert self.model is not None

        acts_dict = {site: [] for site in self.cfg.model_cfg.acts_cfg.sites}
        with self.autocast_context():
            with torch.inference_mode():
                trng = tqdm.trange(0, tokens.shape[0], llm_batch_size, leave=False)
                trng.set_description(f"Tracing {tokens.shape[0]}")
                for i in trng:
                    batch_kwargs = {
                        k: v[i : i + llm_batch_size] for k, v in batched_kwargs.items()
                    }
                    model = self.model
                    d = {}
                    with model.trace(
                        tokens[i : i + llm_batch_size],
                        **self.cfg.model_cfg.model_kwargs,
                        **batch_kwargs,
                    ):
                        for site in self.cfg.model_cfg.acts_cfg.sites:
                            acts_module = getsite(model, site)
                            acts = acts_module.save()
                            d[site] = acts

                    for site in self.cfg.model_cfg.acts_cfg.sites:
                        acts_dict[site].append(d[site])

        acts = {}
        for site in self.cfg.model_cfg.acts_cfg.sites:
            acts[site] = torch.cat(acts_dict[site], dim=0)
        # ### END FUTURE
        acts = DictBatch(data=acts)

        if self.cfg.model_cfg.acts_cfg.force_cast_dtype is not None:
            acts = acts.to(self.cfg.model_cfg.acts_cfg.force_cast_dtype)
        toks_re = tokens
        if self.cfg.model_cfg.acts_cfg.excl_first and not skip_exclude:
            acts = acts[:, 1:]
            toks_re = toks_re[:, 1:]
        if not rearrange:
            assert force_not_skip_padding or not self.cfg.model_cfg.acts_cfg.filter_pad
            return acts
        acts = acts.einops_rearrange("batch seq d_data -> (batch seq) d_data")
        toks_re = einops.rearrange(
            toks_re,
            "batch seq -> (batch seq)",
        )
        if (
            self.cfg.model_cfg.acts_cfg.filter_pad
            and self.model.tokenizer.pad_token_id is not None
        ):
            assert isinstance(self.model.tokenizer.pad_token_id, int)
            mask = toks_re != self.model.tokenizer.pad_token_id
            if not mask.all():
                print(f"removing {(~mask).sum()} activations from pad token locations")
                return acts[mask]
        return acts

    def acts_generator(
        self,
        split: SplitConfig,
        batch_size,
        target_sites: list[str],
        input_sites: list[str],
        nsteps=None,
        id=None,
        nw=None,
        prog_bar=False,
    ):
        chunk_paths = self.cfg._act_chunks_paths(split)
        for i, chunk_path in enumerate(chunk_paths):
            if not chunk_path.exists():
                self._store_chunk(i, chunk_path, split)

        if not (id == nw == None or id is not None and nw is not None):
            raise ValueError("id and nw must be either both None or both not None")
        id = id or 0
        nw = nw or 1
        # g = piler.batch_generator(batch_size, yield_dicts=False)
        # g = piler.batch_generator(batch_size, yield_dicts=False, id=id, nw=nw)
        # next(g)

        print("CHUNK PATHS", chunk_paths[0])

        pilers = [self.cfg.acts_piler(chunk_path, split) for chunk_path in chunk_paths]

        for piler in pilers:
            batch_gen = piler.batch_generator(
                batch_size,
                yield_dicts=False,
                id=id,
                nw=nw,
            )

            for batch in batch_gen:
                yield SAETrainBatch(
                    data=batch.data,
                    input_sites=input_sites,
                    target_sites=target_sites,
                )


class ActsDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for activations
    """

    def __init__(
        self,
        acts: ActsData,
        split: SplitConfig,
        batch_size,
        input_sites: list[str],
        target_sites: list[str],
    ):
        self.acts = acts
        self.split = split
        self.batch_size = batch_size
        self.input_sites = input_sites
        self.target_sites = target_sites

    def store_if_not_exists(self):
        if not self.acts.cfg._acts_piles_path(self.split).exists():
            self.acts._store_split(self.split)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return self.acts.acts_generator(
                self.split,
                self.batch_size,
                input_sites=self.input_sites,
                target_sites=self.target_sites,
            )
        batches_per_pile = (
            self.acts.cfg.generation_config.acts_per_pile // self.batch_size
        )
        id = worker_info.id
        nw = worker_info.num_workers
        assert id % nw == id, (id, nw)
        if self.acts.cfg.databuffer_worker_offset_mult is None:
            offset = (id * batches_per_pile) // nw
        else:
            offset = id * self.acts.cfg.databuffer_worker_offset_mult
        base_size = (
            self.acts.cfg.databuffer_worker_queue_base_size
            if self.acts.cfg.databuffer_worker_queue_base_size is not None
            else int(4096 / self.batch_size * 8 + 1)
        )
        return bufferized_iter(
            self.acts.acts_generator(
                self.split,
                self.batch_size,
                input_sites=self.input_sites,
                target_sites=self.target_sites,
                id=id,
                nw=nw,
            ),
            queue_size=base_size + offset,
            # getnext=lambda i: next(i).cuda(non_blocking=True),
            # getnext=lambda i: next(i).share_memory_(),
        )
