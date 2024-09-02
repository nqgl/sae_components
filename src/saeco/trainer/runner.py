from .saved_model_source_info import ModelReloadInfo
from .run_config import RunConfig
from saeco.initializer.initializer_config import InitConfig
from .train_config import TrainConfig
from .trainable import Trainable

from .trainer import Trainer
from typing import TypeVar
from saeco.architectures.outdated.gate_hierarch import (
    hierarchical_softaux,
    HierarchicalSoftAuxConfig,
    HGatesConfig,
)
from saeco.initializer import Initializer
from saeco.trainer.normalizers import (
    ConstL2Normalizer,
    GeneralizedNormalizer,
)
from saeco.misc.lazy import defer_to_and_set, lazyprop
from saeco.components.resampling.anthropic_resampling import (
    AnthResampler,
)
from functools import cached_property, cache
from pathlib import Path


class TrainingRunner:
    def __init__(self, cfg: RunConfig, model_fn, state_dict=None):
        self.cfg = cfg
        self.model_fn = model_fn
        self._models = None
        self._losses = None
        self._state_dict = state_dict
        self._trainable_loaded = False

    @property
    def state_dict(self):
        return self._state_dict

    @state_dict.setter
    def state_dict(self, value):
        assert self._state_dict is None and self._trainable_loaded is False
        self._state_dict = value

    @cached_property
    def model_name(self):
        return self.model_fn.__name__

    @cached_property
    def name(self):
        return f"{self.model_name}{self.cfg.train_cfg.lr}"

    @cached_property
    def data(self) -> iter:
        return iter(self.cfg.train_cfg.data_cfg.get_databuffer())

    @cached_property
    def initializer(self) -> Initializer:
        return Initializer(
            self.cfg.init_cfg.d_data,
            dict_mult=self.cfg.init_cfg.dict_mult,
            l0_target=self.cfg.train_cfg.l0_target,
            # median=getmed(buf=self.buf, normalizer=self.normalizer),
            # weight_scale=2,
        )

    @defer_to_and_set("_model_fn_output")
    def get_model_fn_output(self):
        assert self._models is self._losses is None
        if self.cfg.arch_cfg is None:
            return self.model_fn(self.initializer)
        return self.model_fn(self.initializer, self.cfg.arch_cfg)

    @cached_property
    def models(self):
        models, losses = self.get_model_fn_output()
        return models

    @cached_property
    def losses(self):
        models, losses = self.get_model_fn_output()
        return losses

    @cached_property
    def trainable(self) -> Trainable:
        self._trainable_loaded = True
        trainable = Trainable(
            self.models,
            self.losses,
            normalizer=self.normalizer,
            resampler=self.resampler,
        ).cuda()
        if self.state_dict is not None:
            load_result = trainable.load_state_dict(self.state_dict)
            print("loaded state dict into trainable:", load_result)
        return trainable

    @cached_property
    def resampler(self) -> AnthResampler:
        res = AnthResampler(self.cfg.resampler_config)
        res.assign_model(
            self.models[0]
        )  # TODO not a big fan of this. maybe just remove the assigning model part of resample class
        return res

    @lazyprop
    def normalizer(self):
        # normalizer = NORMALIZERS[self.cfg.sae_cfg.normalizer]()
        normalizer = GeneralizedNormalizer(
            init=self.initializer, cfg=self.cfg.normalizer_cfg
        )

        normalizer.prime_normalizer(self.data)
        return normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value
        self._normalizer.prime_normalizer(self.data)

    @cached_property
    def trainer(self):
        trainer = Trainer(
            self.cfg.train_cfg,
            run_cfg=self.cfg,
            model=self.trainable,
            wandb_run_label=self.name,
            reload_info=ModelReloadInfo.from_model_fn(self.model_fn),
        )
        # trainer.post_step()
        return trainer

    @classmethod
    def load_saved(
        cls,
        cfg,
        model_fn,
        name,
        modify_cfg_fn=lambda x: x,
        modify_state_dict_fn=lambda x: x,
        models_dir: Path = Path.home() / "workspace/saved_models/",
    ) -> "TrainingRunner":
        import torch

        pt_path = models_dir / (name + ".pt")
        cfg_path = models_dir / (name + ".json")
        cfg = modify_cfg_fn(cfg.model_validate_json(cfg_path.read_text()))
        state = modify_state_dict_fn(torch.load(pt_path))
        tr = TrainingRunner(cfg, model_fn, state_dict=state)
        # tr.trainable.load_state_dict(state)
        return tr

    @classmethod
    def autoload(
        cls,
        name,
        modify_cfg_fn=lambda x: x,
        modify_state_dict_fn=lambda x: x,
        models_dir: Path = Path.home() / "workspace/saved_models/",
    ) -> "TrainingRunner":
        modelpath: Path = models_dir / (name + ".json")
        reload_info = ModelReloadInfo.model_validate_json(
            modelpath.with_suffix(".reload_info.json").read_text()
        )
        model_fn, cfg = reload_info.get_model_and_cfg()
        return cls.load_saved(
            cfg,
            model_fn,
            name,
            modify_cfg_fn=modify_cfg_fn,
            modify_state_dict_fn=modify_state_dict_fn,
            models_dir=models_dir,
        )


def main():
    l0_target = 45
    PROJECT = "nn.Linear Check"
    tcfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 3e-4,
            "L2_loss": 1,
        },
        lr=1e-3,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
        batch_size=2048,
        use_lars=True,
        betas=(0.9, 0.99),
    )
    cfg = RunConfig(
        train_cfg=tcfg,
        arch_cfg=HierarchicalSoftAuxConfig(
            hgates=HGatesConfig(
                l1_scale_base=1,
                num_levels=2,
                BF=2**4,
                untied=False,
                classic=True,
                penalize_inside_gate=False,
                target_hierarchical_l0_ratio=0.5,
                relu_gate_encoders=False,
            )
        ),
        init_cfg=InitConfig(normalizer="L2Normalizer"),
    )

    tr = TrainingRunner(cfg, hierarchical_softaux)

    tr.normalizer = ConstL2Normalizer()
    tr.trainer
    tr.trainer.train()


if __name__ == "__main__":
    main()


def norm_consts(mean, std, geo_med, std_from_med, elementwise_std=False):
    return mean, std, geo_med, std_from_med


"""
if we have a non-elementwise normalizer NE and an elementwise normalizer E
is E(x) = NE(E(x))?

LinLearnedLN(x) = (x - mean) / estd


also is LLLN(x) = T^-1(LLLN(T(x))) for linear transformations T?

"""
