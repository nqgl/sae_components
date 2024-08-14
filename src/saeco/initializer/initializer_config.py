from saeco.misc.lazy import lazyprop
from saeco.sweeps import SweepableConfig


class InitConfig(SweepableConfig):
    d_data: int = 768
    dict_mult: int = 8

    @lazyprop
    def d_dict(self):
        return self.d_data * self.dict_mult

    @d_dict.setter
    def d_dict(self, value):
        assert self.dict_mult is None
        setattr(self, "_d_dict", value)
