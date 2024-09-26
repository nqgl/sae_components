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

if TYPE_CHECKING:
    from saeco.data.dataset import DataConfig


class ActsData:
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
        ):
            acts_piler.distribute(acts)
        acts_piler.shuffle_piles()

    def acts_generator_from_tokens_generator(self, tokens_generator, llm_batch_size):
        for tokens in tokens_generator:
            acts = self.to_acts(tokens, llm_batch_size=llm_batch_size)
            yield acts

    def to_acts(self, tokens, llm_batch_size, rearrange=True, skip_exclude=False):
        acts_list = []
        with torch.autocast(device_type="cuda"):
            with torch.inference_mode():
                for i in range(
                    0,
                    tokens.shape[0],
                    llm_batch_size,
                ):
                    model: nnsight.LanguageModel = self.model
                    with model.trace(
                        tokens[i : i + llm_batch_size],
                        **self.cfg.model_cfg.model_kwargs,
                    ):
                        acts_module = getsite(model, self.cfg.model_cfg.acts_cfg.site)
                        acts = acts_module.save()
                        acts_module.stop()
                    acts_list.append(acts.value)
        acts = torch.cat(acts_list, dim=0).half()
        toks_re = tokens
        if self.cfg.model_cfg.acts_cfg.excl_first and not skip_exclude:
            acts = acts[:, 1:]
            toks_re = toks_re[:, 1:]
        if rearrange:
            acts = einops.rearrange(
                acts,
                "batch seq d_data -> (batch seq) d_data",
            )
            toks_re = einops.rearrange(
                toks_re,
                "batch seq -> (batch seq)",
            )
        if self.cfg.model_cfg.acts_cfg.filter_pad:
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
                print("get next pile")
                print(id, nw, p)
                pile = piler[p]
                if progress is None:
                    progress = tqdm.trange(
                        pile.shape[0] * piler.num_piles // batch_size
                    )

                assert pile.dtype == torch.float16
                print("got next pile")
                for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                    yield pile[i : i + batch_size]
                    progress.update()
        else:
            assert nsteps is None
            # pile = piler[id]
            spares = []
            nspare = 0
            for p in range(id % nw, piler.num_piles, nw):
                print("get next pile")
                print(id, nw, p)
                pile = piler[p]
                # if p == id:
                #     nextpile = nextpile[: (id % nw + 1) * len(nextpile) // nw]
                # pile = nextpile
                assert pile.dtype == torch.float16
                print("got next pile")
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
            print("get next pile")
            print(id, nw, p)
            nextpile = piler[p]
            assert pile.dtype == torch.float16
            print("got next pile")
            yield pile
            pile = nextpile
        for i in range(0, len(pile) // batch_size * batch_size, batch_size):
            yield pile


class ActsDataset(torch.utils.data.IterableDataset):
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
        bpp = self.acts.cfg.generation_config.acts_per_pile // self.batch_size
        id = worker_info.id
        nw = worker_info.num_workers
        offset = (id % nw) * bpp // nw
        return bufferized_iter(
            gen_fn(
                self.split,
                self.batch_size,
                id=id,
                nw=nw,
            ),
            queue_size=32 + offset,
            # getnext=lambda i: next(i).share_memory_(),
        )
