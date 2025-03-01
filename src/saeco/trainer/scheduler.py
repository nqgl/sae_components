from saeco.sweeps import SweepableConfig

# from .schedule_cfg
from .tosteps_wrapper import RunFloat, ResFloat


class Schedule(SweepableConfig):
    schedule: dict[int | float, float]


class RunSchedule(SweepableConfig):
    schedule: dict[int | RunFloat, float]


class ResampleSchedule(SweepableConfig):
    schedule: dict[int | ResFloat, float]


class Scheduler:
    def __init__(
        self,
        schedule: Schedule,
    ): ...
