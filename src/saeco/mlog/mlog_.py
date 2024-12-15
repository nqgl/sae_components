from .fns import logger_instance


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
        logger_instance.init(project=project, config=config, run_name=run_name)

    @staticmethod
    def finish():
        logger_instance.finish()

    @staticmethod
    def log(data: dict, step=None):
        logger_instance.log(data, step=step)

    @staticmethod
    def update_config(config_dict):
        logger_instance.update_config(config_dict)

    @staticmethod
    def config():
        return logger_instance.config_get()

    @staticmethod
    def begin_sweep(sweep_config, project):
        return logger_instance.sweep(sweep_config, project=project)

    @staticmethod
    def run_agent(sweep_id, project, function):
        logger_instance.agent(sweep_id, project=project, function=function)

    @staticmethod
    def enter(project=None, config=None, run_name=None):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            mlog.init(project=project, config=config, run_name=run_name)
            yield
            mlog.finish()

        return ctx()
