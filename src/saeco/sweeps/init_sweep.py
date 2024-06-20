# %%
from typing_extensions import Unpack
from pydantic._internal._generics import PydanticGenericMetadata
import wandb
from typing import TypeVar, Generic, Type, List, Annotated, Union
from pydantic import BaseModel, create_model, dataclasses

PROJECT = "sweep_test"


def initialize_sweep(d):
    sweep_id = wandb.sweep(
        sweep=d,
        project=PROJECT,
    )
    f = open("sweep/sweep_id.txt", "w")
    f.write(sweep_id)
    f.close()


# %%
def run():
    wandb.init()
    scfg = ConfigFromSweep(**wandb.config)
    cfg, lgcfg = get_configs_from_sweep(scfg=scfg)
    # if cfg.sae_cfg.sae_type != "VanillaSAE":
    #     cfg.neuron_dead_threshold = -1
    cfg.l1_coeff = cfg.l1_coeff * sparsity_coeff_adjustment(scfg)
    wandb.config.update({"adjusted_l1_coeff": cfg.l1_coeff})

    nice_name = wandb.config["sae_type"]
    wandb.finish()

    l0_target = 45
    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 3e-4,
            "L2_loss": 10,
        },
        lr=1e-3,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
        batch_size=2048,
        use_lars=True,
        betas=(0.9, 0.99),
    )


# from torch.utils.viz._cycles import warn_tensor_cycles
#
# warn_tensor_cycles()

from dataclasses import dataclass, field

sweep_id = open("sweeps/sweep_id.txt").read().strip()

SWEEP_NAME = "sweep_test"


def main():
    wandb.agent(
        sweep_id,
        function=run,
        project=PROJECT,
    )


if __name__ == "__main__":
    main()

# %%
