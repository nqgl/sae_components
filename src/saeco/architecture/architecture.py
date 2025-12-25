import typing
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from types import get_original_bases
from typing import Any, Literal, overload

import torch
from paramsight import get_resolved_typevars_for_base, takes_alias
from torch import nn

import saeco.components as co
import saeco.core as cl
from saeco.architecture.arch_prop import (
    aux_model_prop,
    loss_prop,
    metric_prop,
    model_prop,
)
from saeco.architecture.arch_reload_info import ArchStoragePaths
from saeco.components.losses import Loss
from saeco.components.metrics.metrics import ActMetrics, PreActMetrics
from saeco.components.resampling.anthropic_resampling import AnthResampler
from saeco.components.resampling.freq_tracker.ema import EMAFreqTracker
from saeco.initializer.initializer import Initializer
from saeco.misc.utils import useif
from saeco.sweeps import SweepableConfig
from saeco.trainer.evaluation_protocol import ReconstructionEvaluatorFunctionProtocol
from saeco.trainer.normalizers.normalizer import StaticInvertibleGeneralizedNormalizer
from saeco.trainer.run_config import RunConfig
from saeco.trainer.trainable import Trainable
from saeco.trainer.trainer import Trainer


class SAE(cl.Seq):
    encoder_pre: cl.Module
    preacts: PreActMetrics
    nonlinearity: cl.Module
    encoder: cl.Module
    decoder: cl.Module
    acts: ActMetrics
    freqs: EMAFreqTracker | None
    penalty: co.Penalty | None
    losses: list[Loss]

    @overload
    def __init__(
        self,
        *,
        encoder_pre: Literal[None] = None,
        nonlinearity: Literal[None] = None,
        encoder: nn.Module,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...

    @overload
    def __init__(
        self,
        *,
        encoder_pre: nn.Module,
        nonlinearity: nn.Module,
        encoder: Literal[None] = None,
        decoder: nn.Module,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ): ...
    def __init__(
        self,
        *,
        encoder_pre: nn.Module | None = None,
        nonlinearity: nn.Module | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        act_metrics: ActMetrics | None = None,
        preacts: PreActMetrics | None = None,
        penalty: co.Penalty | None = ...,
        freqs: EMAFreqTracker | None = ...,
    ):
        penalty = co.L1Penalty() if penalty is ... else penalty
        freqs = EMAFreqTracker() if freqs is ... else freqs
        act_metrics = ActMetrics() if act_metrics is None else act_metrics
        preacts = PreActMetrics() if preacts is None else preacts
        # we could seek a preactmetrics on the encoder in the future
        assert (encoder_pre is None) == (nonlinearity is None)
        assert (encoder is None) != (nonlinearity is None)
        if encoder is None:
            encoder = cl.Seq(
                encoder_pre=encoder_pre,
                preacts=preacts,
                nonlinearity=nonlinearity,
            )
        super().__init__(
            # normalizer=normalizer,
            # encoder_pre=encoder_pre,
            # preacts=preacts,
            # nonlinearity=nonlinearity,
            encoder=encoder,
            acts=act_metrics,
            **useif(freqs is not None, freqs=freqs),
            **useif(penalty is not None, penalty=penalty),
            decoder=decoder,
            # denormalizer=...
        )

    def set_to_aux_model(self, aux_name: str):
        self.acts.name = aux_name

    # @cached_property
    # def encode(self) -> cl.Seq:
    #     return self[:"acts"]

    # @cached_property
    # def decode(self) -> cl.Seq:
    #     return self["decoder":]

    # # encode: cl.Module = NotImplemented
    # @classmethod
    # def build_model(
    #     cls,
    #     normalizer: StaticInvertibleGeneralizedNormalizer,
    #     encoder_pre: cl.Module,
    #     nonlinearity: cl.Module,
    #     decoder: cl.Module,
    # ):
    #     return cls(
    #         encoder_pre=encoder_pre,
    #         nonlinearity=nonlinearity,
    #         decoder=decoder,
    #     )


# class NormalizedModel(cl.Module):
#     def __init__(self, normalizer, model: SAE) -> None:
#         super().__init__()
#         self.normalizer = normalizer


# CFG_PATH_EXT = ".arch_cfg"
# ARCH_CLASS_REF_PATH_EXT = ".arch_ref"


class Architecture[ArchConfigT: SweepableConfig]:
    def __init__(
        self,
        run_cfg: "RunConfig[ArchConfigT]",
        state_dict: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
    ):
        self.run_cfg: RunConfig[ArchConfigT] = run_cfg
        self.state_dict: dict[str, Any] | None = state_dict
        self._instantiated: bool = False
        self._setup_complete: bool = False
        self._trainable: Trainable | None = None
        self.device: torch.device | str = device

    @property
    def cfg(self) -> ArchConfigT:
        return self.run_cfg.arch_cfg

    def instantiate(self, inst_cfg: dict[str, Any] | None = None):
        if inst_cfg:
            self.run_cfg = self.run_cfg.from_selective_sweep(inst_cfg)
        assert self.cfg.is_concrete()
        assert self.run_cfg.is_concrete()
        self._instantiated = True
        self._setup()

    def _setup(self):
        assert self._instantiated
        self.setup()
        self._setup_complete = True

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
            #         f"aux_models must be a list of SAEs or lists of SAEs, got {type(e)}"
            #     )
        return l

    @abstractmethod
    def setup(self): ...

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
    def data(
        self,
    ):  # TODO maybe this should be called dataloader and return dataloader, unless important to not reuse datapoints
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

    @takes_alias
    @classmethod
    def get_arch_config_class(cls):
        if cls is Architecture:
            raise ValueError(
                "Architecture class must not be generic to get config class"
            )
        bases = get_original_bases(cls)
        assert len(bases) == 1
        p = typing.get_args(bases[0])
        assert len(p) == 1
        return p[0]

    @takes_alias
    @classmethod
    def get_config_class(cls):
        return RunConfig[cls.get_arch_config_class()]

    def save_to_path(
        self,
        path: Path | ArchStoragePaths,
        save_weights: bool = ...,
        averaged_weights: bool | None = None,
    ):
        if isinstance(path, Path):
            path = ArchStoragePaths.from_path(path)
        path.path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            self.save_to_path(path.path.with_name(f"{path.path.name}_1"))
            raise ValueError(
                f"file already existed at {path}, wrote to {path.path.name}_1"
            )

        from .arch_reload_info import ArchRef

        arch_ref = ArchRef.from_arch(self)

        path.arch_ref.write_text(arch_ref.model_dump_json())
        if save_weights is True or (
            save_weights is ... and self._trainable is not None
        ):
            if self._trainable is None:
                raise ValueError("trainable is None but attempted to save weights")
            torch.save(self._trainable.state_dict(), path.model_weights)  # type: ignore
        if averaged_weights is not None:
            torch.save(averaged_weights, path.averaged_weights)  # type: ignore
        return path

    # @classmethod
    # def load_from_path(cls, path: Path | ArchStoragePaths, load_weights=None):
    #     from .arch_reload_info import ArchClassRef

    #     path = ArchStoragePaths.from_path(path)
    #     if cls is Architecture:
    #         arch_ref = ArchClassRef.model_validate_json(path.arch_cls_ref.read_text())
    #         arch_cls = arch_ref.get_arch_class()
    #         return arch_cls.load_from_path(path, load_weights=load_weights)
    #     cfg_cls = cls.get_config_class()
    #     cfg = cfg_cls.model_validate_json(path.cfg.read_text())
    #     state_dict = None
    #     if path.model_weights.exists():
    #         if load_weights is None:
    #             raise ValueError(
    #                 f"weights exist at {path.model_weights}, but load_weights is not set"
    #             )
    #         if load_weights:
    #             state_dict = torch.load(path.model_weights)
    #     else:
    #         if load_weights:
    #             raise ValueError(
    #                 f"weights do not exist at {path.model_weights}, but load_weights is set"
    #             )
    #     inst = cls(cfg, state_dict=state_dict)
    #     if cfg.is_concrete():
    #         inst.instantiate()
    #     return inst

    @classmethod
    def load(
        cls,
        path: Path | ArchStoragePaths,
        load_weights: bool | None = None,
        averaged_weights: bool | None = False,
    ):
        return ArchStoragePaths.from_path(path).load_arch(
            load_weights=load_weights, averaged_weights=averaged_weights
        )

    def run_training(self):
        self.trainer.train()

    def get_sweep_manager(self, ezpod_group=None):
        from saeco.sweeps.newsweeper import SweepManager

        return SweepManager(self, ezpod_group=ezpod_group)

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


class BaseRunConfig[ArchConfigT: SweepableConfig](SweepableConfig):
    arch_cfg: ArchConfigT


class WithConfig[ArchConfigT: SweepableConfig]:
    cfg: ArchConfigT


class ArchitectureBase[ArchConfigT: SweepableConfig]:
    """ """

    def __init__(
        self,
        run_cfg: BaseRunConfig[ArchConfigT],
        state_dict: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
    ):
        assert state_dict is None
        self.run_cfg: BaseRunConfig[ArchConfigT] = run_cfg
        # self.state_dict: dict[str, Any] | None = state_dict
        self._instantiated: bool = False
        self._setup_complete: bool = False
        # self._trainable: Trainable | None = None
        # self.device: torch.device | str = device

    @property
    def cfg(self) -> ArchConfigT:
        return self.run_cfg.arch_cfg

    def instantiate(self, inst_cfg: dict[str, Any] | None = None):
        if inst_cfg:
            self.run_cfg = self.run_cfg.from_selective_sweep(inst_cfg)
        assert self.cfg.is_concrete() and self.run_cfg.is_concrete()
        self._instantiated = True
        self._setup()

    def _setup(self):
        assert self._instantiated
        self.setup()
        self._setup_complete = True

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
            #         f"aux_models must be a list of SAEs or lists of SAEs, got {type(e)}"
            #     )
        return l

    @abstractmethod
    def setup(self): ...

    @takes_alias
    @classmethod
    def get_arch_config_class(cls) -> type[ArchConfigT]:
        if cls is ArchitectureBase:
            raise ValueError(
                "Architecture class must not be generic to get config class"
            )
        return get_resolved_typevars_for_base(cls, ArchitectureBase)[0]

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type[BaseRunConfig[ArchConfigT]]: ...

    # return BaseRunConfig[cls.get_arch_config_class()]

    def save_to_path(
        self,
        path: Path | ArchStoragePaths,
        save_weights: bool = ...,
        averaged_weights: bool | None = None,
    ):
        if isinstance(path, Path):
            path = ArchStoragePaths.from_path(path)
        path.path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            self.save_to_path(
                path.path.with_name(f"{path.path.name}_1"),
                save_weights=save_weights,
                averaged_weights=averaged_weights,
            )
            raise ValueError(
                f"file already existed at {path}, wrote to {path.path.name}_1"
            )

        from .arch_reload_info import ArchRef

        arch_ref = ArchRef.from_arch(self)

        path.arch_ref.write_text(arch_ref.model_dump_json())
        if save_weights is True or (
            save_weights is ... and self._trainable is not None
        ):
            if self._trainable is None:
                raise ValueError("trainable is None but attempted to save weights")
            torch.save(self._trainable.state_dict(), path.model_weights)  # type: ignore
        if averaged_weights is not None:
            torch.save(averaged_weights, path.averaged_weights)  # type: ignore
        return path

    @classmethod
    def load(
        cls,
        path: Path | ArchStoragePaths,
        load_weights: bool | None = None,
        averaged_weights: bool | None = False,
    ) -> "ArchitectureBase":
        return ArchStoragePaths.from_path(path).load_arch(
            load_weights=load_weights, averaged_weights=averaged_weights, xcls=cls
        )

    @abstractmethod
    def run_training(self): ...

    def get_sweep_manager(self, ezpod_group=None):
        from saeco.sweeps.newsweeper import SweepManager

        return SweepManager(self, ezpod_group=ezpod_group)

    def save_sweepref_and_get_py_commands(
        self,
        project: str,
        gpus_per_run: int,
        clivars: str = 'TORCH_LOGS="graph_breaks,recompiles"  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ',
        pre_commands: str = "",
        pyname: str | None = None,
    ) -> list[str]:
        sm = self.get_sweep_manager()
        sm.initialize_sweep(project=project)

        commands = (
            sm.get_worker_run_commands_for_manual_sweep(suffix="--distributed-skip-log")
            if gpus_per_run > 1
            else sm.get_worker_run_commands_for_manual_sweep()
        )
        return to_py_cmd(
            commands,
            pyname=pyname
            or ("python3" if gpus_per_run == 1 else f"composer -n {gpus_per_run}"),
            challenge_file=None,
            prefix_vars=pre_commands + clivars,
        )


def to_py_cmd(  # Semi temporary code duplication of something that also exists in ezpod
    cmd: str | list[str],
    pyname: str,
    challenge_file: str | None = None,
    prefix_vars: str | None = None,
):
    if isinstance(cmd, list):
        return [
            to_py_cmd(
                c, pyname=pyname, challenge_file=challenge_file, prefix_vars=prefix_vars
            )
            for c in cmd
        ]
    cmd = f"{pyname} {cmd}"
    if prefix_vars:
        cmd = f"{prefix_vars} {cmd}"
    if challenge_file:
        cmd = f"{pyname} {challenge_file}; {cmd}"
    return cmd
