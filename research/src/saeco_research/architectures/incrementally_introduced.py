import torch
import torch.nn as nn

import saeco.components as co
from saeco.components import (
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
)
from saeco.components.penalties import L1PenaltyScaledByDecoderNorm
from saeco.core import Seq
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.initializer import Initializer
from saeco.misc import useif
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig


class Config(SweepableConfig):
    pre_bias: bool = False
    clip_grad: float | None = Swept[float | None](0.1, 0.01, 0.001, 0.003)


def sae(
    init: Initializer,
    cfg: Config,
):
    # init._encoder.add_wrapper(ReuseForward)
    dec_mul_l1 = L1PenaltyScaledByDecoderNorm()
    init._decoder.const_init_bias()
    init._encoder.const_init_bias()
    model_full = Seq(
        **useif(cfg.pre_bias, pre_bias=init._decoder.sub_bias()),
        encoder=Seq(linear=init.encoder.resampled(), relu=nn.ReLU()),
        freqs=EMAFreqTracker(),
        l1=dec_mul_l1,
        metrics=co.metrics.ActMetrics(),
        decoder=dec_mul_l1.set_decoder(init.decoder.resampled()),
    )

    models = [model_full]
    losses = dict(
        L2_loss=L2Loss(model_full),
        sparsity_loss=SparsityPenaltyLoss(model_full),
    )

    return models, losses


from saeco.data import DataConfig, ModelConfig
from saeco.sweeps import do_sweep
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.runner import TrainingRunner

model_fn = sae

PROJECT = "sae sweeps"
quick_check = False
train_cfg = TrainConfig(
    data_cfg=DataConfig(
        model_cfg=ModelConfig(acts_cfg=ActsDataConfig(excl_first=not quick_check))
    ),
    raw_schedule_cfg=RunSchedulingConfig(
        run_length=80_000,
        resample_period=10_000,
        resample_delay=45_000,
        resampling_finished_phase=0.15,
        lr_cooldown_length=0.1,
        targeting_post_resample_hiatus=0,
        targeting_delay=15_000,
        targeting_post_resample_step_size_warmup=0.2,
        # resample_delay=0.69,
        lr_warmup_length=0,
    ),
    l0_target=25,
    coeffs={
        "sparsity_loss": 4e-3,
        "L2_loss": 1,
    },
    lr=3e-4,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.0003,
    batch_size=4096,
    use_lars=True,
    betas=Swept[tuple[float, float]]((0.9, 0.999)),
)
acfg = Config(
    pre_bias=Swept[bool](False),
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)

runcfg = RunConfig[Config](
    train_cfg=train_cfg,
    arch_cfg=acfg,
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(
            optim_momentum=(0.0),
            dec_momentum=Swept(False),
            bias_momentum=Swept(0.0),
            b2_technique=Swept("sq"),  # sq better
            b2_scale=Swept(1.0),
        ),
        bias_reset_value=-0.02,
        enc_directions=0,
        dec_directions=1,
        freq_balance=25,
    ),
)


class FreqBalanceSweep(SweepableConfig):
    run_cfg: RunConfig[Config] = runcfg
    # target_l0: int = Swept(2)
    # target_l0: int = Swept(2, 3, 5, 15, 25, 35, 50)
    target_l0: int | None = Swept(5, 25)
    target_l1: int | float | None = Swept(1, 5)
    n: int = Swept(0, 4, 6, 8, 10)
    num_steps: int = 4000
    noise_scale: float = Swept(1, 0.1, 0.01, 0.003)


cfg: FreqBalanceSweep = FreqBalanceSweep()

# cfg = cfg.random_sweep_configuration()


def run(cfg: FreqBalanceSweep):
    tr = TrainingRunner(cfg.run_cfg, model_fn=sae)
    t = tr.trainer
    assert tr.normalizer.primed
    tr.resampler.assign_model(tr.trainable)
    tr.resampler.wholistic_freqbalance(
        model=tr.trainable,
        datasrc=tr.data,
        target_l0=cfg.target_l0,
        target_l1=cfg.target_l1,
    )
    dictrange = torch.arange(cfg.run_cfg.init_cfg.d_dict)

    features = [
        f
        for fd in [
            tr.trainable.model.model.module.encoder.linear.features,
            tr.trainable.model.model.module.decoder.features,
        ]
        for f in fd.values()
    ]
    valid = dictrange % (2**cfg.n) == 0
    with torch.no_grad():
        for i, f in enumerate(features):
            f.features[~valid] = 0
    for i in range(1, cfg.n):
        tr.trainer.train(num_steps=cfg.num_steps)
        n = 2 ** (cfg.n - i)
        valid = dictrange % (2 * n) == 0
        new_valid = (dictrange + n) % (2 * n) == 0
        with torch.no_grad():
            rand = None
            for f in features:
                f.features[valid] /= 2
                f.features[new_valid] = f.features[valid]
                if f.type == "bias":
                    continue
                fn = f.features[valid].norm(dim=1, keepdim=True)
                if rand is None:
                    rand = (
                        1 - 0.5 * torch.rand_like(f.features[valid])
                    ) * cfg.noise_scale
                    nf = f.features[valid] / fn
                    rand -= (rand * nf).sum(dim=1, keepdim=True) * nf
                    assert (rand * nf).sum(dim=1).abs().max() < 1e-6
                f.features[new_valid] += rand * fn
                f.features[valid] -= rand * fn
    tr.trainer.train()


if __name__ == "__main__":
    do_sweep(True, "rand" if quick_check else None)
