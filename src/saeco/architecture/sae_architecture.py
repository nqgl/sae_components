import typing
from functools import cached_property
from typing import Any

import torch
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import nn

from saeco.architecture.arch_prop import (
    aux_model_prop,
    loss_prop,
    metric_prop,
    model_prop,
)
from saeco.architecture.architecture import ArchitectureBase
from saeco.architecture.sae import SAE
from saeco.components.losses import Loss
from saeco.components.resampling.anthropic_resampling import AnthResampler
from saeco.initializer.initializer import Initializer
from saeco.sweeps import SweepableConfig
from saeco.trainer.evaluation_protocol import ReconstructionEvaluatorFunctionProtocol
from saeco.trainer.normalizers.normalizer import StaticInvertibleGeneralizedNormalizer
from saeco.trainer.run_config import RunConfig
from saeco.trainer.trainable import Trainable
from saeco.trainer.trainer import Trainer


class Architecture[ArchConfigT: SweepableConfig](
    ArchitectureBase[RunConfig[ArchConfigT]]
):
    """Base class for an SAE architecture.

    Subclass it, parameterized by your config type, and declare the model
    and losses with the property decorators:

      - ``@model_prop`` — the method that builds the core SAE
      - ``@loss_prop`` — a training loss
      - ``@aux_model_prop`` — an auxiliary model with its own losses
      - ``setup()`` — optional hook to attach layer mixins before the
        model is built

    From those declarations you get the training pipeline, save/load,
    sweep enumeration, and resampling for free::

        class VanillaSAE(Architecture[VanillaConfig]):
            @model_prop
            def model(self):
                return SAE(encoder=self.init.encoder, decoder=self.init.decoder)

            @loss_prop
            def L2_loss(self):
                return L2Loss(self.model)

        arch = VanillaSAE(run_cfg)
        arch.run_training()

    ``self.cfg`` is the architecture config; ``self.init`` is the
    ``Initializer`` (parameter factories sized from ``init_cfg``).
    """

    # run_cfg: RunConfig[ArchConfigT]

    def __init__(
        self,
        run_cfg: RunConfig[ArchConfigT],
        state_dict: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
    ):
        super().__init__(run_cfg, state_dict=state_dict, device=device)
        self.run_cfg = run_cfg
        self._trainable: Trainable | None = None

    @property
    def cfg(self) -> ArchConfigT:
        return self.run_cfg.arch_cfg

    @takes_alias
    @classmethod
    def get_arch_config_class(cls):
        if cls is Architecture:
            raise ValueError(
                "Architecture class must not be generic to get config class"
            )
        config_class = get_resolved_typevars_for_base(cls, Architecture)[0]
        if isinstance(config_class, typing.TypeVar):
            raise ValueError(
                "Architecture class must not be generic to get config class"
            )
        return config_class

    @cached_property
    def _core_model(self) -> SAE:
        return model_prop.get_from_fields(self)

    @cached_property
    def _losses(self) -> dict[str, Loss]:
        return loss_prop.get_from_fields(self)

    @cached_property
    def _metrics(self) -> dict[str, nn.Module]:
        return metric_prop.get_from_fields(self)

    @cached_property
    def aux_models(self) -> list[SAE]:
        aux_models: dict[str, SAE] = aux_model_prop.get_from_fields(self)
        l: list[SAE] = []
        for name, e in aux_models.items():
            # if isinstance(e, list):
            #     # e: list[SAE]
            #     for i, sae in enumerate(e):
            #         sae.set_to_aux_model(f"{name}.{i}")
            #         l.append(sae)
            # elif isinstance(e, SAE):
            e: SAE
            e.set_to_aux_model(name)
            l.append(e)
            # else:
            #     raise ValueError(
            #         f"aux_models must be a list of SAEs or lists of "
            #         f"SAEs, got {type(e)}"
            #     )
        return l

    @property
    def trainable(self) -> Trainable:
        if self._trainable is None:
            self._trainable = self.make_trainable().to(device=self.device)
        return self._trainable

    def make_trainable(self):
        if not self._instantiated:
            self.instantiate()
        trainable = Trainable(
            [self._core_model, *self.aux_models],
            losses=self._losses,
            normalizer=self.normalizer,
            resampler=self.make_resampler(),
        )
        if self.state_dict is not None:
            load_result = trainable.load_state_dict(self.state_dict)
            print("loaded state dict into trainable:", load_result)
        return trainable

    def make_resampler(self):
        resampler = AnthResampler(self.run_cfg.resampler_config)
        resampler.assign_model(self._core_model)
        return resampler

    @cached_property
    def init(self) -> Initializer:
        return Initializer(
            self.run_cfg.init_cfg.d_data,
            dict_mult=self.run_cfg.init_cfg.dict_mult,
            l0_target=self.run_cfg.train_cfg.l0_target,
        )

    @cached_property
    def data(self):
        # TODO maybe this should be called dataloader and return
        # dataloader, unless important to not reuse datapoints
        return iter(self.run_cfg.train_cfg.get_databuffer())

    @cached_property
    def normalizer(self) -> StaticInvertibleGeneralizedNormalizer:
        normalizer = StaticInvertibleGeneralizedNormalizer(
            init=self.init, cfg=self.run_cfg.normalizer_cfg
        ).to(device=self.device)
        if self.state_dict is None:
            normalizer.prime_normalizer(self.data)
        return normalizer

    @cached_property
    def trainer(self):
        trainer = Trainer(
            self.run_cfg.train_cfg,
            run_cfg=self.run_cfg,
            model=self.trainable,
            save_callback=self.save_to_path,
            recons_eval_fns=self.get_evaluation_functions(),
        )
        return trainer

    def _save_weights_by_default(self) -> bool:
        return self._trainable is not None

    def _state_dict_for_save(self) -> dict[str, Any]:
        if self._trainable is None:
            raise ValueError("trainable is None but attempted to save weights")
        return self._trainable.state_dict()

    def run_training(self):
        self.trainer.train()

    def get_evaluation_functions(
        self,
    ) -> dict[str, ReconstructionEvaluatorFunctionProtocol]:
        from saeco.trainer.recons import (
            get_recons_loss_no_bos,
            get_recons_loss_with_bos,
        )

        return {
            "recons/with_bos/": get_recons_loss_with_bos,
            "recons/no_bos/": get_recons_loss_no_bos,
        }
