from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import (
    gemma_2_2b_openwebtext,
)
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig, TrainingRunner
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

if __name__ == "__main__":
    from saeco.sweeps import do_sweep

    do_sweep(True)
else:
    from .model import AnthUpdateConfig, anth_update_model

    PROJECT = "L0Targeting_cmp"
    cfg = RunConfig[AnthUpdateConfig](
        train_cfg=TrainConfig(
            data_cfg=gemma_2_2b_openwebtext,
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=17_000,
                resample_period=4_000,
                lr_end_plateau_length=2_000,
            ),
            #
            batch_size=4096,
            optim="Adam",
            lr=Swept(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
            # lr=Swept(3e-4, 6e-4, 1e-3, 2e-3, 3e-3, 6e-3, 1e-2, 1e-2),
            # lr=1e-3,
            betas=(0.9, 0.999),
            #
            use_autocast=False,
            use_lars=False,
            #
            l0_targeting_enabled=False,
            l0_target=25.45,
            l0_target_adjustment_size=0.0002,
            coeffs={
                "sparsity_loss": Swept(
                    *[1e-3 * 1.2 ** (i * 2) for i in range(6)]
                ),  # 3e-3 gets like 50, 1e-2 gets like 5
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
        init_cfg=InitConfig(dict_mult=8, d_data=2304),
        arch_cfg=AnthUpdateConfig(),
    )


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=anth_update_model)
    tr.trainer.train()
