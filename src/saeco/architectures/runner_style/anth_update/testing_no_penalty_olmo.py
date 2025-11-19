from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.split_config import SplitConfig
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

    PROJECT = "L0Targeting"
    cfg = RunConfig[AnthUpdateConfig](
        train_cfg=TrainConfig(
            data_cfg=DataConfig(
                dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
                model_cfg=ModelConfig(
                    acts_cfg=ActsDataConfig(
                        excl_first=True,
                        site="transformer.h.20.input",
                        d_data=4096,
                    ),
                    model_name="allenai/OLMo-2-1124-7B",
                ),
                trainsplit=SplitConfig(start=0, end=5, tokens_from_split=10_000_000),
                generation_config=DataGenerationProcessConfig(
                    # tokens_per_pile=2**25,
                    acts_per_pile=2**16,
                    meta_batch_size=2**18,
                    llm_batch_size=2**10,
                ),
                seq_len=256,
            ),
            raw_schedule_cfg=RunSchedulingConfig(
                run_length=50_000,
                resample_period=9_000,
                lr_warmup_length=2_000,
                lr_cooldown_length=0.2,
                targeting_warmup_length=0,
                lr_geometric_rescale=True,
                lr_cooldown_factor=0.03,
                lr_resample_warmup_factor=0.03,
                lr_warmup_factor=0.01,
            ),
            #
            batch_size=4096,
            optim="Adam",
            lr=Swept(5e-4, 1e-3, 2e-3),
            # lr=Swept(3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
            betas=Swept[tuple[float, float]]((0.9, 0.999), (0.90237, 0.99872)),
            # betas=Swept[tuple[float, float]]((0.8147611221, 0.98123126)),
            #
            use_autocast=False,
            use_lars=False,
            #
            l0_targeting_enabled=True,
            l0_target=50,
            l0_target_adjustment_size=0.0007,
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
            optim_reset_cfg=OptimResetValuesConfig(),
        ),
        #
        init_cfg=InitConfig(),
        arch_cfg=AnthUpdateConfig(clip_grad=1.0),
    )


def run(cfg):
    tr = TrainingRunner(cfg, model_fn=anth_update_model)
    tr.trainer.train()
