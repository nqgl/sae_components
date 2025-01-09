from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import ActsDataConfig, DataConfig, ModelConfig
from saeco.data.data_config_definitions import gpt_2_block
from saeco.initializer import InitConfig
from saeco.sweeps import SweepableConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig, TrainingRunner
from saeco.trainer.run_config import RunConfig
from saeco.trainer.tosteps_wrapper import RunFloat
from saeco.trainer.train_config import TrainConfig


if __name__ == "__main__":
    from saeco.sweeps import do_sweep

    do_sweep(True)
else:
    from .model import anth_update_model, AnthUpdateConfig

    PROJECT = "L0Targeting_cmp"
    cfg = RunConfig[AnthUpdateConfig](
        train_cfg=TrainConfig(
            data_cfg=gpt_2_block(layer=6),
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=50_000,
                resample_period=9_000,
                lr_end_plateau_length=1_000,
                targeting_pre_deflation=Swept(0.0, 0.1),
            ),
            #
            batch_size=4096,
            optim="Adam",
            # lr=Swept(3e-4, 6e-4, 1e-3, 2e-3, 3e-3, 6e-3, 1e-2, 1e-2),
            lr=1e-3,
            betas=(0.9, 0.999),
            #
            use_autocast=False,
            use_lars=False,
            #
            l0_targeter_type=Swept("gentle_basic", "basic", "pid"),
            l0_targeting_enabled=True,
            l0_target=Swept[float](
                10.852739,
                8.054066,
                56.044189,
            ),
            l0_target_adjustment_size=Swept(1e-4, 3e-4),
            coeffs={
                # "sparsity_loss": 5e-3,
                "sparsity_loss": Swept(1e-3, 3e-3),
                # Swept(
                #     3e-4, 1e-3, 3e-3, 5e-3, 8e-3, 1.1e-2
                # ),  # 3e-3 gets like 50, 1e-2 gets like 5
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
        arch_cfg=AnthUpdateConfig(),
    )


def run(cfg):

    tr = TrainingRunner(cfg, model_fn=anth_update_model)
    tr.trainer.train()
