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
from saeco.trainer.train_config import TrainConfig


if __name__ == "__main__":
    from saeco.sweeps import do_sweep

    do_sweep(True)
else:
    from .model import anth_update_model, AnthUpdateConfig

    PROJECT = "L0Targeting"
    cfg = RunConfig[AnthUpdateConfig](
        train_cfg=TrainConfig(
            data_cfg=gpt_2_block(),
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=50_000,
                resample_period=12_500,
            ),
            #
            batch_size=4096,
            optim="Adam",
            lr=1e-3,
            betas=(0.9, 0.999),
            #
            use_autocast=True,
            use_lars=True,
            #
            l0_target=Swept[float](30.3),  # , 86.5, 308.8),
            l0_target_adjustment_size=Swept(0.0003, 0.0001, 0.001, 0.003),
            l0_targeter_type=Swept("pid", "basic", "gentle_basic"),
            l0_targeting_enabled=True,
            coeffs={
                "sparsity_loss": Swept(3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                "L2_loss": 1,
            },
            #
            wandb_cfg=dict(project=PROJECT),
            checkpoint_period=5000,
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
