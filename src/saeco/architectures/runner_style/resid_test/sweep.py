from saeco.architectures.anth_update import model

from saeco.sweeps import do_sweep

model_fn = model.sae
PROJECT = "sae sweeps"

if __name__ == "__main__":
    do_sweep(True)
