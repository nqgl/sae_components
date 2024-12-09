import saeco.core.module as cl
from typing import Callable, List
from saeco.components import Loss
from saeco.initializer.initializer import Initializer
from saeco.sweeps.sweepable_config import SweepableConfig


class Model(cl.Module):
    def __init__(self, loss_cls_list: List[Loss]):
        super().__init__()
        self.losses = ...


from attrs import define, field


@define
class Architecture:
    model_gen_fn: Callable[
        [Initializer, SweepableConfig], tuple[list[cl.Module], dict[str, Loss]]
    ]
    base_cfg: SweepableConfig

    def instantiated_config_from_sweep_dict(self, cfg_info: dict):
        return self.base_cfg.from_selective_sweep(cfg_info)

    def run(self, cfg: SweepableConfig):
        tr = TrainingRunner(cfg, model_fn=self.model_gen_fn)
        tr.trainer.train()
