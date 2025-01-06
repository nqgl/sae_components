from saeco.architecture.arch_reload_info import ArchRef
from saeco.mlog import mlog


from attrs import define, field

from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData


@define
class SweepRunner:
    root_arch_ref: ArchRef
    sweep_id: str | None = field(default=None)

    def run_sweep(self):
        with mlog.enter():
            arch = self.root_arch_ref.load_arch()
            arch.instantiate(mlog.config())
            mlog.update_config(full_cfg=arch.run_cfg.model_dump())
            arch.run_training()
        return arch

    def run_random_instance(self):
        arch = self.root_arch_ref.load_arch()
        cfg: SweepableConfig = self.root_arch_ref.config
        arch.instantiate(cfg.to_swept_nodes().random_selection())
        with mlog.enter():
            mlog.update_config(full_cfg=arch.run_cfg.model_dump())
            arch.run_training()
        return arch

    def start_sweep_agent(self):
        mlog.start_sweep_agent(self.sweep_id, self.run_sweep)

    @classmethod
    def from_sweepdata(cls, sweep_data: "SweepData"):
        return cls(root_arch_ref=sweep_data.root_arch_ref, sweep_id=sweep_data.sweep_id)
