from saeco.architectures.anth_update import anth_update_model
from saeco.sweeps import do_sweep
from saeco.trainer.runner import TrainingRunner


def run(cfg):
    tr = TrainingRunner(cfg.run_cfg, model_fn=anth_update_model)
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True)
