import torch

# from saeco.architectures.basic import sae, TrainingRunner, cfg
import torch.nn as nn

from saeco.architectures.initialization.initializer import Initializer
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)

from saeco.core.reused_forward import ReuseForward
from saeco.core import Seq
import saeco.components.features.features as ft

import saeco.components as co
from saeco.misc import useif
from saeco.sweeps import SweepableConfig


class Config(SweepableConfig):
    pre_bias: bool = False
    untied: bool = True
    freq_balance_l0_target: int = 45


def sae(
    init: Initializer,
    cfg: Config,
):
    # init._encoder.add_wrapper(ReuseForward)
    if cfg.untied:
        init._decoder.add_wrapper(ft.NormFeatures)
        init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    else:
        init._decoder.tie_weights(init._encoder)

    def model(enc, penalties, metrics, detach=False):
        return Seq(
            **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
            encoder=enc,
            freqs=EMAFreqTracker(),
            **penalties,
            metrics=metrics,
            decoder=init.decoder.resampled(),
        )

    model_full = model(
        enc=Seq(linear=init.encoder.resampled(), relu=nn.ReLU()),
        penalties=dict(l1=L1Penalty()),
        metrics=co.metrics.ActMetrics(),
        detach=False,
    )

    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses


from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.data import DataConfig, ModelConfig, ActsDataConfig
from saeco.sweeps import Swept, do_sweep


model_fn = sae

PROJECT = "sae sweeps"
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=False))
    ),
    l0_target=45,
    coeffs={
        "sparsity_loss": 5e-3,
        "L2_loss": 1,
    },
    lr=7e-4,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.001,
    batch_size=2048,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.99)),
    schedule.resample_period=1000,
)
acfg = Config(
    pre_bias=Swept[bool](True),
)
cfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
)


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=model_fn)
    tr.trainer.train()


@torch.no_grad()
def pre_freqbalance(model, datasrc, bias, freqs, target_l0):
    original_beta = freqs.beta
    freqs.beta = 0.9
    for i in range(50):
        lr = 75 / (1.03**i)
        if i < 10:
            lr /= 2 ** (10 - i)
        for _ in range(10):
            with torch.autocast("cuda"):
                d = next(datasrc)
                model(d)
            target_freq = target_l0 / freqs.freqs.shape[0]
            fn = lambda x: x
            diff = fn(torch.tensor(target_freq)) - fn(freqs.freqs)
            bias.data += diff * lr
        print(i, diff.abs().mean().item(), freqs.freqs.sum())
        # freqs.reset()
    freqs.beta = original_beta


def weird_l0_target(model, datasrc, encoder, freqs, target_l0):
    bias = encoder.linear.bias
    freqs.beta = 0.9

    @torch.no_grad()
    def wrapper():
        # print("wrapper called")
        target_freq = target_l0 / freqs.freqs.shape[0]
        fn = lambda x: x
        undershoot = torch.relu(target_l0 - freqs.freqs.sum())
        diff = fn(
            torch.tensor(target_freq / (1 + 10 * (1 - undershoot / (1 + undershoot))))
            - fn(freqs.freqs)
        )
        sub = torch.relu(freqs.freqs.sum() - target_l0)
        bias.data += torch.relu(diff * 250)
        bias.data -= torch.where(diff < 0, sub / 100, 0)

    assert not hasattr(encoder, "post_step_hook")
    setattr(encoder, "post_step_hook", wrapper)
    assert hasattr(encoder, "post_step_hook")


trainable_here = []
cfg = cfg.random_sweep_configuration()

tr = TrainingRunner(cfg, model_fn=sae)
pre_freqbalance(
    model=tr.trainable,
    datasrc=tr.buf,
    bias=tr.trainable.model.model.module.encoder.linear.bias,
    freqs=tr.trainable.model.model.module.freqs,
    target_l0=4,
)
# weird_l0_target(
#     model=tr.trainable,
#     datasrc=tr.buf,
#     encoder=tr.trainable.model.model.module.encoder,
#     freqs=tr.trainable.model.model.module.freqs,
#     target_l0=45,
# )
tr.trainer.train()
# if __name__ == "__main__":

#     do_sweep(True, "rand")
