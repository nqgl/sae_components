import torch.utils
from sae_components.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, Optional
from sae_components.components.losses import L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass, field
from sae_components.trainer.train_cache import TrainCache
from sae_components.trainer.trainable import Trainable
from sae_components.trainer.post_backward_normalization import (
    do_post_backward,
    do_post_step,
)
from .recons import get_recons_loss
from transformer_lens import HookedTransformer
from sae_components.data.sc.dataset import DataConfig, SplitConfig, TokensData


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class ModelConfig:
    layer: int = 6
    model_name: str = "gpt2-small"
    site: str = "resid-pre"
    # d_data: int = 768
    # expansion_factor: int = 8

    def __post_init__(self):
        model = None

        def getmodel():
            nonlocal model
            if model is None:
                model = HookedTransformer.from_pretrained(self.model_name)
            return model

        self._getmodel = getmodel

    @property
    def model(self) -> HookedTransformer:
        return self._getmodel()


@dataclass
class TrainConfig:
    coeffs: dict[str, float] = field(default_factory=lambda: dict(sparsity_loss=1e-3))
    # optim_config: OptimConfig = OptimConfig()
    l0_target: Optional[float] = None
    l0_target_adjustment_size: float = 0.0003
    use_autocast: bool = True
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    batch_size: int = 4096
    wandb_cfg: dict = field(default_factory=dict(project="sae-components"))
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Trainable,
        namestuff=None,
        optim: torch.optim.Optimizer = None,
    ):
        self.cfg = cfg
        self.model = model
        # self.sae.provide("optim", self.optim)
        wandb.init(
            **cfg.wandb_cfg,
            config={"model": repr(model), "cfg": cfg},
            # entity="sae_all",
            reinit=True,
        )
        if namestuff is not None:
            wandb.run.name = (
                f"{namestuff}[{cfg.l0_target}]-{wandb.run.name.split('-')[-1]}"
            )
        self.t = 1
        self.extra_calls = []
        self.optim = optim or torch.optim.RAdam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
        )

        self.llm_val_tokens = TokensData(
            self.cfg.data_cfg, self.cfg.model_cfg.model
        ).get_tokens_from_split(self.cfg.data_cfg.testsplit)
        self.intermittent_metric_freq = 1000
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None

    def post_backward(self):
        do_post_backward(self.model)

    def post_step(self):
        do_post_step(self.model)

    def log(self, d):
        if wandb.run is not None:
            wandb.log(d, step=self.t)

    def coeffs(self):
        # TODO
        return self.cfg.coeffs

    def proc_cache_after_forward(self, cache: TrainCache):
        if self.cfg.l0_target is not None:
            self.cfg.coeffs["sparsity_loss"] = self.cfg.coeffs["sparsity_loss"] * (
                1
                + (-1 if self.cfg.l0_target > cache.L0 else 1)
                * self.cfg.l0_target_adjustment_size
            )
            self.log({"dynamic_sparsity_coeff": self.cfg.coeffs["sparsity_loss"]})

    def get_databuffer(self, num_batches=None, num_workers=0):
        ds = self.cfg.data_cfg.train_dataset(
            self.cfg.model_cfg.model, batch_size=self.cfg.batch_size
        )
        return torch.utils.data.DataLoader(ds, num_workers=num_workers)

        return self.cfg.data_cfg.train_data_batch_generator(
            model=self.cfg.model_cfg.model, batch_size=4096, nsteps=num_batches
        )

    def train(self, buffer=None):
        if buffer is None:
            buffer = self.get_databuffer(num_workers=0)
            buffer = iter(buffer)
            # buffer = self.cfg.data_cfg.train_data_batch_generator(
            #     model=self.cfg.model_cfg.model, batch_size=self.cfg.batch_size
            # )
        if self.t <= 1:
            self.model.normalizer.prime_normalizer(buffer)
        self.post_step()

        def tocuda(buffer):
            for bn in buffer:
                yield bn.cuda().squeeze(0)

        def bufferize_fn(buffer):
            def buf_fn():
                p = next(buffer)
                for bn in buffer:
                    # bn = bn.cuda()
                    yield p
                    p = bn

            return buf_fn()

        buffer = tocuda(buffer)
        # for _ in range(100):
        #     buffer = bufferize_fn(buffer)

        for bn in buffer:
            # bn = bn.cuda()
            self.optim.zero_grad()
            if isinstance(bn, tuple):
                x, y = bn
            else:
                x = bn
                y = x

            cache = TrainCache()
            if self.cfg.use_autocast:
                with torch.autocast(device_type="cuda"):
                    loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
                    self.proc_cache_after_forward(cache)

            else:
                loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
                self.proc_cache_after_forward(cache)

            # self.proc_cache_after_forward(cache)
            if self.cfg.use_autocast:
                self.gradscaler.scale(loss).backward()
            else:
                loss.backward()
            self.post_backward()

            if self.cfg.use_autocast:
                self.gradscaler.step(self.optim)
                self.gradscaler.update()
            else:
                self.optim.step()
            self.post_step()
            self.full_log(cache)
            self.t += 1
            # print(f"loss: {loss.item()}")
            del cache.forward_reuse_dict
            cache.destroy_children()
            del cache
            if self.t % self.intermittent_metric_freq == 0:
                self.log_recons("recons/with_bos/", True)
                self.log_recons("recons/no_bos/", False)
            # for key in [k for k in cache.forward_reuse_dict.keys()]:
            #     del cache.forward_reuse_dict[key]
            #     del key
            # del x, y, loss, cache

    def log_recons(self, label, proc_bos, num_batches=5):
        self.log(
            {
                (label + k): v
                for k, v in get_recons_loss(
                    self.cfg.model_cfg.model,
                    self.model,
                    buffer=None,
                    all_tokens=self.llm_val_tokens,
                    cfg=self.cfg.data_cfg.acts_config,
                    bos_processed_with_hook=proc_bos,
                    num_batches=num_batches,
                ).items()
            }
        )

    def full_log(self, cache: Cache):
        if self.t % 10 != 0:
            return
        # d = cache.logdict(
        #     excluded=[
        #         "acts",
        #         "y_pred",
        #         "x",
        #         "y",
        #         "resample",
        #         "nonlinear_argsmaxed",
        #         "acts_spoof",
        #     ]
        # )
        if wandb.run is not None:
            wandb.log(cache.logdict(), step=self.t)

    def save(self):
        torch.save(
            self.sae.state_dict(), "/root/workspace/" + wandb.run.name + f"_{self.t}.pt"
        )
