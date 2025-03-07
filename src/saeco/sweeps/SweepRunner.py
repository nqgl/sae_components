from saeco.mlog import mlog


from attrs import define, field

from saeco.sweeps.sweepable_config.sweepable_config import SweepableConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData


@define
class SweepRunner:
    sweep_data: "SweepData"
    sweep_index: int | None = field(default=None)
    sweep_hash: str | None = field(default=None)

    @property
    def run_name(self):
        run_name = f"{self.sweep_data.root_arch_ref.class_ref.cls_name}"
        if self.sweep_index is not None:
            run_name = f"{run_name}_{self.sweep_index}"
        return f"{self.sweep_data.sweep_id}:{run_name}"

    def run_sweep(self):
        with mlog.enter(
            arch_ref=self.sweep_data.root_arch_ref,
            project=self.sweep_data.project,
            run_name=self.run_name,
        ):

            arch = self.sweep_data.root_arch_ref.load_arch()
            cfg = mlog.config()
            arch.instantiate(cfg)
            mlog.update_config(full_cfg=arch.run_cfg.model_dump())
            self.log_sweep_info(cfg)
            arch.run_training()
        return arch

    def log_sweep_info(self, cfg: SweepableConfig):
        mlog.log_sweep(
            cfg,
            sweep_data=self.sweep_data,
            sweep_index=self.sweep_index,
            sweep_hash=self.sweep_hash,
        )

    def run_random_instance(self):
        arch = self.sweep_data.root_arch_ref.load_arch()
        cfg: SweepableConfig = self.sweep_data.root_arch_ref.config
        arch.instantiate(cfg.to_swept_nodes().random_selection())
        with mlog.enter(
            arch_ref=self.sweep_data.root_arch_ref,
            run_name=self.run_name,
        ):
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

    # def run_local_sweep(self):
    #     if self.sweep_index is not None:
    #         kwargs = {"sweep_index": self.sweep_index, "sweep_hash": self.sweep_hash}
    #         mlog.use_custom_sweep()

    #     else:
    #         kwargs = {}
    #     mlog.start_sweep_agent(self.sweep_data, self.run_sweep, **kwargs)

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
