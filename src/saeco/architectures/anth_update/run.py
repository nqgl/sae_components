from saeco.architectures.anth_update import model_fn
from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg.run_cfg, model_fn=model_fn)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
