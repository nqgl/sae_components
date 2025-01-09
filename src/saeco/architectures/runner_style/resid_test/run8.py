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
    from .model import resid_sae, ResidConfig, ReuseLayer

    PROJECT = "L0Targeting"
    cfg = RunConfig[ResidConfig](
        train_cfg=TrainConfig(
            data_cfg=gpt_2_block(layer=6),
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=50_000,
                resample_period=9_000,
                # lr_warmup_length=2000,
                # lr_cooldown_length=0.2,
                # lr_geometric_rescale=True,
                # lr_cooldown_factor=0.03,
                # lr_resample_warmup_factor=0.03,
                # lr_warmup_factor=0.01,
            ),
            #
            batch_size=4096,
            optim="Adam",
            lr=1e-3,
            # lr=Swept(3e-4, 1e-3),
            betas=(0.9, 0.997),
            # betas=Swept[tuple[float, float]]((0.8147611221, 0.98123126)),
            #
            use_autocast=False,
            use_lars=True,
            #
            l0_targeting_enabled=True,
            l0_target=50,
            l0_target_adjustment_size=0.0003,
            coeffs={
                "sparsity_loss": 3e-3,  # 3e-3 gets like 50, 1e-2 gets like 5
                "L2_loss": 1,
            },
            #
            wandb_cfg=dict(project=PROJECT),
            # old img runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
            checkpoint_period=None,
        ),
        resampler_config=AnthResamplerConfig(
            optim_reset_cfg=OptimResetValuesConfig(), enc_directions=2
        ),
        #
        init_cfg=InitConfig(),
        arch_cfg=ResidConfig(
            layers=4,
            pre_bias=False,
            individual_dec_bias=True,
            individual_enc_bias=True,
            anth_scale=False,
            thresh_range=(1, 0.5),
            # Swept[tuple[float, float]](
            #     (0, 0),
            #     (1, 0),
            #     (0, 1),
            #     (1, 1),
            #     (0.5, 0.5),
            #     (0.5, 0),
            #     (0, 0.5),
            # ),
            reuse_layer=1,
            unpenalized_restricted_final=False,
        ),
    )


def run(cfg):

    tr = TrainingRunner(cfg, model_fn=resid_sae)
    tr.trainer.train()
