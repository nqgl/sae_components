from saeco.sweeps import SweepableConfig


class InitConfig(SweepableConfig):
    """Sizing for the dictionary.

    ``d_data`` is the input/residual dimension; ``dict_mult`` is the
    expansion factor, so the dictionary size is
    ``d_dict = int(d_data * dict_mult)``.
    """

    d_data: int = 768
    dict_mult: int | float = 8

    @property
    def d_dict(self):
        return int(self.d_data * self.dict_mult)
