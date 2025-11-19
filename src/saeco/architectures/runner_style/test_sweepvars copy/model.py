




from saeco.sweeps import SweepableConfig


class Config(SweepableConfig):
    a: int
    b: int
    a_b: int


from saeco.sweeps import do_sweep


def run(cfg):
    print(cfg)


if __name__ == "__main__":
    do_sweep(True, "new run")
else:
    pass
