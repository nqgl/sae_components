from contextlib import contextmanager
from typing import TYPE_CHECKING

import einops
import nnsight
import torch
import tqdm
from nnsight import LanguageModel

from saeco.data.bufferized_iter import bufferized_iter
from saeco.data.split_config import SplitConfig
from saeco.data.tokens_data import TokensData
from saeco.misc.nnsite import getsite
from saeco.misc import str_to_dtype
import torch.utils
import torch.utils.data

if TYPE_CHECKING:
    from saeco.data.data_cfg import DataConfig


class ActsData:
    """
    Generates, stores, and loads activations
    """

    def __init__(self, cfg: "DataConfig", model: LanguageModel | None):
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
            position=0,
            desc="Storing",
        ):
            acts_piler.distribute(acts)
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
    ):
        assert isinstance(tokens, torch.Tensor)

        ### CURRENT
        acts_list = []
        with self.autocast_context():
            with torch.inference_mode():
                trng = tqdm.trange(0, tokens.shape[0], llm_batch_size, leave=False)
                trng.set_description(f"Tracing {tokens.shape[0]}")
                for i in trng:
                    batch_kwargs = {
                        k: v[i : i + llm_batch_size] for k, v in batched_kwargs.items()
                    }
                    model = self.model
                    with model.trace(
                        tokens[i : i + llm_batch_size],
                        **self.cfg.model_cfg.model_kwargs,
                        **batch_kwargs,
                    ):
                        acts_module = getsite(model, self.cfg.model_cfg.acts_cfg.site)
                        acts = acts_module.save()
                        # stopsite = getsite(
                        #     model,
                        #     self.cfg.model_cfg.acts_cfg.site.replace("input", "output"),
                        # )
                        # stopsite.stop()
                        # TODO: add back stopping when available
                    if not isinstance(acts, torch.Tensor):
                        raise ValueError(f"acts is not a torch.Tensor: {type(acts)}")
                    acts_list.append(acts)
        acts = torch.cat(acts_list, dim=0)
        ### END CURRENT

        # ### FUTURE

        # acts_dict = {site: [] for site in self.cfg.model_cfg.acts_cfg.sites}
        # with self.autocast_context():
        #     with torch.inference_mode():
        #         trng = tqdm.trange(0, tokens.shape[0], llm_batch_size, leave=False)
        #         trng.set_description(f"Tracing {tokens.shape[0]}")
        #         for i in trng:
        #             batch_kwargs = {
        #                 k: v[i : i + llm_batch_size] for k, v in batched_kwargs.items()
        #             }
        #             model = self.model
        #             d = {}
        #             with model.trace(
        #                 tokens[i : i + llm_batch_size],
        #                 **self.cfg.model_cfg.model_kwargs,
        #                 **batch_kwargs,
        #             ):
        #                 for site in self.cfg.model_cfg.acts_cfg.sites:
        #                     acts_module = getsite(model, site)
        #                     acts = acts_module.save()
        #                     d[site] = acts

        #             for site in self.cfg.model_cfg.acts_cfg.sites:
        #                 acts_dict[site].append(d[site])

        # acts = {}
        # for site in self.cfg.model_cfg.acts_cfg.sites:
        #     acts[site] = torch.cat(acts_dict[site], dim=0)
        # ### END FUTURE

        if self.cfg.model_cfg.acts_cfg.force_cast_dtype is not None:
            acts = acts.to(self.cfg.model_cfg.acts_cfg.force_cast_dtype)
        toks_re = tokens
        if self.cfg.model_cfg.acts_cfg.excl_first and not skip_exclude:
            acts = acts[:, 1:]
            toks_re = toks_re[:, 1:]
        if not rearrange:
            assert force_not_skip_padding or not self.cfg.model_cfg.acts_cfg.filter_pad
            return acts
        acts = einops.rearrange(
            acts,
            "batch seq d_data -> (batch seq) d_data",
        )
        toks_re = einops.rearrange(
            toks_re,
            "batch seq -> (batch seq)",
        )
        if (
            self.cfg.model_cfg.acts_cfg.filter_pad
            and self.model.tokenizer.pad_token_id is not None
        ):
            mask = toks_re != self.model.tokenizer.pad_token_id
            if not mask.all():
                print(f"removing {(~mask).sum()} activations from pad token locations")
                return acts[toks_re != self.model.tokenizer.pad_token_id]
        return acts

    def acts_generator(
        self,
        split: SplitConfig,
        batch_size,
        nsteps=None,
        id=None,
        nw=None,
        prog_bar=False,
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

        if prog_bar:
            print("\nProgress bar activation batch count is approximate\n")

            progress = (
                tqdm.trange(nsteps or split.tokens_from_split // batch_size)
                if split.tokens_from_split is not None
                else None
            )
            for p in range(id % nw, piler.num_piles, nw):
                # print("get next pile")
                # print(id, nw, p)
                pile = piler[p]
                if progress is None:
                    progress = tqdm.trange(
                        pile.shape[0] * piler.num_piles // batch_size
                    )

                assert pile.dtype == torch.float16
                # print("got next pile")
                for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                    yield pile[i : i + batch_size]
                    progress.update()
        else:
            assert nsteps is None
            # pile = piler[id]
            spares = []
            nspare = 0
            for p in range(id % nw, piler.num_piles, nw):
                # print("get next pile")
                # print(id, nw, p)
                pile = piler[p]
                # if p == id:
                #     nextpile = nextpile[: (id % nw + 1) * len(nextpile) // nw]
                # pile = nextpile
                assert pile.dtype == self.cfg.model_cfg.acts_cfg.storage_dtype
                # print("got next pile")
                for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                    yield pile[i : i + batch_size]
                spare = pile[len(pile) // batch_size * batch_size :]
                if len(spare) > 0:
                    spares.append(spare)
                    nspare += len(spare)
                    if nspare > batch_size:
                        consolidated = torch.cat(spares, dim=0)
                        for i in range(
                            0, len(consolidated) // batch_size * batch_size, batch_size
                        ):
                            yield consolidated[i : i + batch_size]
                        spare = consolidated[
                            len(consolidated) // batch_size * batch_size :
                        ]
                        spares = [spare]
                        nspare = len(spare)

            # for i in range(0, len(pile) // batch_size * batch_size, batch_size):
            #     yield pile[i : i + batch_size]

    def pile_generator(
        self,
        split: SplitConfig,
        batch_size,
        nsteps=None,
        id=None,
        nw=None,
        prog_bar=False,
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

        assert nsteps is None
        pile = piler[id]
        for p in range(id + nw, piler.num_piles, nw):
            # print("get next pile")
            # print(id, nw, p)
            nextpile = piler[p]
            # print("got next pile")
            yield pile
            pile = nextpile
        for i in range(0, len(pile) // batch_size * batch_size, batch_size):
            yield pile


class ActsDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for activations
    """

    def __init__(
        self, acts: ActsData, split: SplitConfig, batch_size, generate_piles=False
    ):
        self.acts = acts
        self.split = split
        self.batch_size = batch_size
        self.generate_piles = generate_piles

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.generate_piles:
            gen_fn = self.acts.pile_generator
        else:
            gen_fn = self.acts.acts_generator
        if worker_info is None:
            return gen_fn(self.split, self.batch_size)
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
            gen_fn(
                self.split,
                self.batch_size,
                id=id,
                nw=nw,
            ),
            queue_size=base_size + offset,
            # getnext=lambda i: next(i).cuda(non_blocking=True),
            # getnext=lambda i: next(i).share_memory_(),
        )
