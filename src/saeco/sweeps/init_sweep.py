assert False, "This file is not meant to be imported and is due for deletion"

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
from saeco.trainer import TrainingRunner


def mksweeprun(model_fn): ...


# import mksweeprun
# run = mksweeprun(model_fn)


def mkrun(sweepfile_module):
    def run():
        wandb.init()
        basecfg: BaseModel = sweepfile_module.cfg
        runfn: callable = sweepfile_module.run
        cfg = basecfg.model_validate(wandb.config)
        runfn(cfg)
        wandb.finish()

    return run


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
