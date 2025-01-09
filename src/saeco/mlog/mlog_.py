from saeco.sweeps.sweepable_config.SweptNode import SweptNode
from .fns import logger_instance
import os


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

    def setter(self, __fset):
        raise NotImplementedError

    def __set__(self, owner_self, value):
        raise NotImplementedError


class mlog:
    @staticmethod
    def init(project=None, config=None, run_name=None):
        print("init")
        logger_instance.init(project=project, config=config, run_name=run_name)
        print("init done")

    @staticmethod
    def finish():
        logger_instance.finish()

    @staticmethod
    def log(data: dict, step=None):
        logger_instance.log(data, step=step)

    @staticmethod
    def update_config(**config_dict):
        logger_instance.update_config(config_dict)

    @staticmethod
    def config():
        return logger_instance.config_get()

    @staticmethod
    def create_sweep(swept_nodes: SweptNode, project):
        return logger_instance.sweep(swept_nodes, project=project)

    @staticmethod
    def start_sweep_agent(sweep_id, function):
        logger_instance.agent(sweep_id, function=function)

    @staticmethod
    def enter(project=None, config=None, run_name=None):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            mlog.init(project=project, config=config, run_name=run_name)
            # mlog.update_config(dict(pod_info=mlog._get_pod_info()))
            yield
            mlog.finish()

        return ctx()

    @staticmethod
    def _get_pod_info():
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
