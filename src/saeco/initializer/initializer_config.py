from saeco.misc.lazy import lazyprop
from saeco.sweeps import SweepableConfig


class InitConfig(SweepableConfig):
    d_data: int = 768
    dict_mult: int | float = 8

    @property
    def d_dict(self):
        return int(self.d_data * self.dict_mult)
