from saeco.architecture.arch_reload_info import ArchRef
from saeco.mlog import mlog


from attrs import define, field

from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData


@define
class SweepRunner:
    sweep_data: "SweepData"
    sweep_id: str | None = field(default=None)
    sweep_index: int | None = field(default=None)
    sweep_hash: str | None = field(default=None)

    def run_sweep(self):
        with mlog.enter(arch_ref=self.sweep_data.root_arch_ref):
            print("run_sweep entered")
            arch = self.sweep_data.root_arch_ref.load_arch()
            print("run_sweep instantiated")
            arch.instantiate(mlog.config())
            print("run_sweep updated config")
            mlog.update_config(full_cfg=arch.run_cfg.model_dump())
            print("run_sweep running training")
            arch.run_training()
            print("run_sweep finished training")
        return arch

    def run_random_instance(self):
        arch = self.sweep_data.root_arch_ref.load_arch()
        cfg: SweepableConfig = self.sweep_data.root_arch_ref.config
        arch.instantiate(cfg.to_swept_nodes().random_selection())
        with mlog.enter(arch_ref=self.sweep_data.root_arch_ref):
            mlog.update_config(full_cfg=arch.run_cfg.model_dump())
            arch.run_training()
        return arch

    def start_sweep_agent(self):
        if self.sweep_index is not None:
            kwargs = {"sweep_index": self.sweep_index, "sweep_hash": self.sweep_hash}
            mlog.use_custom_sweep()

        else:
            kwargs = {}
        mlog.start_sweep_agent(self.sweep_data, self.run_sweep, **kwargs)

    @classmethod
    def from_sweepdata(
        cls,
        sweep_data: "SweepData",
        sweep_index: int | None = None,
        sweep_hash: str | None = None,
    ):
        assert (sweep_index is not None) == (sweep_hash is not None)

        return cls(
            sweep_data=sweep_data, sweep_index=sweep_index, sweep_hash=sweep_hash
        )
