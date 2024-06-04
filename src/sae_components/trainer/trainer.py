from sae_components.core import Cache
import torch
import wandb
from typing import Protocol, runtime_checkable, Optional
from sae_components.components.losses import L2Loss, SparsityPenaltyLoss
from dataclasses import dataclass, field
from sae_components.trainer.train_cache import TrainCache
from sae_components.trainer.trainable import Trainable
from sae_components.trainer.post_backward_normalization import post_backward, post_step
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
    optim_config: OptimConfig = OptimConfig()
    l0_target: Optional[float] = None
    l0_target_adjustment_size: float = 0.0003
    use_autocast: bool = False
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    lr: float = 3e-4


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
            project="sae-components",
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
            self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999)
        )

        self.llm_val_tokens = TokensData(
            self.cfg.data_cfg, self.cfg.model_cfg.model
        ).get_tokens_from_split(self.cfg.data_cfg.testsplit)
        self.intermittent_metric_freq = 1000
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None

    def post_backward(self):
        self.model.apply(post_backward)

    def post_step(self):
        self.model.apply(post_step)

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

    def get_databuffer(self, num_batches=None):
        return self.cfg.data_cfg.train_data_batch_generator(
            model=self.cfg.model_cfg.model, batch_size=4096, nsteps=num_batches
        )

    def train(self, buffer=None):
        if buffer is None:
            buffer = self.cfg.data_cfg.train_data_batch_generator(
                model=self.cfg.model_cfg.model, batch_size=4096
            )
        if self.t <= 1:
            self.model.normalizer.prime_normalizer(buffer)
        self.post_step()
        for bn in buffer:
            self.optim.zero_grad()
            if isinstance(bn, tuple):
                x, y = bn
            else:
                x = bn
                y = x

            cache = TrainCache()
            if self.cfg.use_autocast:
                with torch.cuda.amp.autocast():
                    loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())
            else:
                loss = self.model.loss(x, cache=cache, coeffs=self.coeffs())

            self.proc_cache_after_forward(cache)
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
            print(f"loss: {loss.item()}")
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
