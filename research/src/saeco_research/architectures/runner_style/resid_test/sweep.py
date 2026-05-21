from saeco.sweeps import do_sweep
from saeco_research.architectures.anth_update import model

model_fn = model.sae
PROJECT = "sae sweeps"

if __name__ == "__main__":
    do_sweep(True)
