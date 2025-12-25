from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import gpt_2_block
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig, TrainingRunner
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

if __name__ == "__main__":
    from saeco.sweeps import do_sweep

    do_sweep(True)
else:
    from .model import ResidConfig, resid_sae

    PROJECT = "L0Targeting"
    cfg = RunConfig[ResidConfig](
        train_cfg=TrainConfig(
            data_cfg=gpt_2_block(layer=6),
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=50_000,
                resample_period=9_000,
                lr_warmup_length=Swept[int](
                    0, 300, 3000, 10_000
                ),  # Swept[int](0, 100, 300, 2_000, 6_000, 20_000),
                lr_cooldown_length=0.2,
            ),
            #
            batch_size=4096,
            optim="Adam",
            lr=Swept(
                1e-3,
                2e-3,
                3e-3,
                6e-3,
                1e-2,
                15e-3,
                22e-3,
                3e-2,
                4e-2,
                5e-2,
                8e-2,
                1.2e-1,
            ),
            betas=(0.9, 0.999),
            #
            use_autocast=True,
            use_lars=False,
            #
            l0_targeting_enabled=True,
            l0_target=25.45,
            l0_target_adjustment_size=0.0002,
            coeffs={
                "sparsity_loss": 4e-3,  # 3e-3 gets like 50, 1e-2 gets like 5
                "L2_loss": 1,
            },
            #
            wandb_cfg=dict(project=PROJECT),
            # old img runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
            checkpoint_period=None,
        ),
        resampler_config=AnthResamplerConfig(
            optim_reset_cfg=OptimResetValuesConfig(),
        ),
        #
        init_cfg=InitConfig(),
        arch_cfg=ResidConfig(),
    )


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=resid_sae)
    tr.trainer.train()
