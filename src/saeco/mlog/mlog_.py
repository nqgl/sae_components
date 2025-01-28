from saeco.sweeps.sweepable_config.SweptNode import SweptNode
from .fns import get_logger, CustomSweeper
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saeco.sweeps.SweepRunner import SweepRunner
    from saeco.sweeps.newsweeper import SweepData


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

    def setter(self, __fset):
        raise NotImplementedError

    def __set__(self, owner_self, value):
        raise NotImplementedError


class mlog:
    logger_instance = get_logger()

    @classmethod
    def use_custom_sweep(cls):
        print("using custom sweep logger!")
        if not isinstance(cls.logger_instance, CustomSweeper):
            cls.logger_instance = CustomSweeper(cls.logger_instance)

    @classmethod
    def init(cls, arch_ref=None, project=None, config=None, run_name=None):
        print("init")
        cls.logger_instance.init(project=project, config=config, run_name=run_name)
        print("init done")

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
            # mlog.update_config(dict(pod_info=mlog._get_pod_info()))
            yield
            print("finish run context")
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
