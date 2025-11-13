from saeco.data.config.data_config_definitions import (
    gpt_2_block,
    gemma_2_2b_openwebtext_test,
    gemma_2_2b_openwebtext,
    gemma_2_2b_openwebtext_test_fp16,
    gemma_2_2b_openwebtext_test_fp32,
    gemma_2_2b_openwebtext_fp32,
    gemma_2_2b_openwebtext_bf16,
)
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer.run_config import RunConfig
from saeco.architectures.gated.arch import GatedConfig, Gated
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data import DataConfig, ModelConfig
from saeco.sweeps import SweepableConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.train_config import TrainConfig
from saeco.initializer import InitConfig

PROJECT = "sae sweeps"

cfg = RunConfig[GatedConfig](
    train_cfg=TrainConfig(
        # data_cfg=gemma_2_2b_openwebtext_bf16(17),
        data_cfg=gpt_2_block(),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=48_000,
            resample_period=9_000,
            lr_cooldown_length=0.3,
            lr_warmup_length=500,
        ),
        #
        batch_size=4096 * 2,
        optim="Adam",
        lr=Swept(1e-4, 3e-4, 1e-3, 2e-3),
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 1.1e-3,
            "L2_loss": 1,
        },
        #
        wandb_cfg=dict(project=PROJECT),
        intermittent_metric_freq=1000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=GatedConfig(),
)
# from transformers import Gemma2ForCausalLM

# import saeco.data.model_cfg as mc

# # mc.MODEL_FN_CALLABLE_OVERRIDE = Gemma2ForCausalLM.from_pretrained
g = Gated(cfg)


sweep_manager = g.get_sweep_manager()
sweep_manager.initialize_sweep(project="sweeping-test-gated", custom_sweep=True)
sweep_manager.run_manual_sweep_with_monitoring(
    cfg.to_swept_nodes().swept_combinations_count_including_vars(),
    purge_after=True,
    setup_min=8,
    prefix_vars="USE_NEPTUNE=true ",
)
