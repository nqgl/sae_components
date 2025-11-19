# from saeco.architectures.hierarchical_iterated_frozen import cfg, run
from saeco.sweeps import do_sweep

PROJECT = "sae sweeps"

if __name__ == "__main__":
    do_sweep(True)
else:
    pass
