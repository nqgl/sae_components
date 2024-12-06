import wandb
import comet_ml
from contextlib import contextmanager

WANDB = True
COMET = False
CLEARML = False


def init():
    if WANDB:
        wandb.init()
    if COMET:
        comet_ml.init()
    if CLEARML:
        assert False


def finish():
    if WANDB:
        wandb.finish()
    if COMET:
        comet_ml.end()
    if CLEARML:
        assert False


# sweep
# config
# finish
# agent


@contextmanager
def enter():
    init()
    yield
    finish()
