import os
from typing import TYPE_CHECKING

from saeco.sweeps.sweepable_config.SweptNode import SweptNode

from .fns import NeptuneCustomLogger, NeptuneScaleLogger, WandbCustomLogger

if TYPE_CHECKING:
    from saeco.sweeps.newsweeper import SweepData

DEFAULT_LOGGER = os.environ.get("SAECO_DEFAULT_LOGGER", "neptune")
LOGGER_CLASSES = {
    "wandb": WandbCustomLogger,
    "neptune": NeptuneCustomLogger,
    "neptune_scale": NeptuneScaleLogger,
}

logger_class = LOGGER_CLASSES[DEFAULT_LOGGER]()


class mlog:
    logger_instance: WandbCustomLogger | NeptuneCustomLogger | NeptuneScaleLogger = (
        logger_class
    )

    @classmethod
    def use_neptune_scale(cls):
        if isinstance(cls.logger_instance, NeptuneScaleLogger):
            return
        assert cls.logger_instance is None or cls.logger_instance.run is None
        cls.logger_instance = NeptuneScaleLogger()

    @classmethod
    def init(cls, arch_ref=None, project=None, config=None, run_name=None):
        cls.logger_instance.init(project=project, config=config, run_name=run_name)

    @classmethod
    def finish(cls):
        cls.logger_instance.finish()

    @classmethod
    def log(cls, data: dict, step=None):
        cls.logger_instance.log(data, step=step)

    @classmethod
    def update_config(cls, **config_dict):
        cls.logger_instance.update_config(config_dict)

    @classmethod
    def config(cls):
        return cls.logger_instance.config_get()

    @classmethod
    def create_sweep(cls, swept_nodes: SweptNode, project):
        return cls.logger_instance.sweep(swept_nodes, project=project)

    @classmethod
    def set_project(cls, project):
        cls.logger_instance.project = project

    @classmethod
    def start_sweep_agent(cls, sweep_data: "SweepData", function, **kwargs):
        cls.logger_instance.agent(sweep_data, function=function, **kwargs)

    @classmethod
    def enter(cls, arch_ref=None, project=None, config=None, run_name=None):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            mlog.init(
                arch_ref=arch_ref, project=project, config=config, run_name=run_name
            )
            yield
            mlog.finish()

        return ctx()

    @classmethod
    def _get_pod_info(cls):
        return dict(
            id=os.environ.get("RUNPOD_POD_ID", "local"),
            hostname=os.environ.get("RUNPOD_POD_HOSTNAME", None),
            gpu_count=os.environ.get("RUNPOD_GPU_COUNT", None),
            cpu_count=os.environ.get("RUNPOD_CPU_COUNT", None),
            public_ip=os.environ.get("RUNPOD_PUBLIC_IP", None),
            datacenter_id=os.environ.get("RUNPOD_DC_ID", None),
            volume_id=os.environ.get("RUNPOD_VOLUME_ID", None),
            cuda_version=os.environ.get("CUDA_VERSION", None),
            pytorch_version=os.environ.get("PYTORCH_VERSION", None),
        )

    @classmethod
    def log_sweep(
        cls,
        selective_instance_sweep_dict,
        sweep_data: "SweepData",
        sweep_index,
        sweep_hash,
    ):
        cls.logger_instance.log_sweep(
            selective_instance_sweep_dict=selective_instance_sweep_dict,
            sweep_data=sweep_data,
            sweep_index=sweep_index,
            sweep_hash=sweep_hash,
        )

    @classmethod
    def get_run_name(cls):
        return cls.logger_instance.run_name
