from saeco.architectures.dynamic_thresh_prolu.model import (
    DynamicThreshConfig,
    DynamicThreshSAE,
    ThreshConfig,
)
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_cfg import DataConfig
from saeco.data.config.data_config_definitions import (
    gemma_2_2b_openwebtext_bf16,
    gpt_2_block,
)
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.model_config.model_cfg import ActsDataConfig, ModelConfig
from saeco.data.config.split_config import SplitConfig
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig


def gpt_2(block_postfix):
    return DataConfig(
        dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        model_cfg=ModelConfig(
            acts_cfg=ActsDataConfig(
                excl_first=True,
                sites=(
                    [f"transformer.h.{bp}" for bp in block_postfix]
                    if isinstance(block_postfix, list | tuple)
                    else [f"transformer.h.{block_postfix}"]
                ),
                d_data=768,
                autocast_dtype_str="bfloat16",
                force_cast_dtype_str="bfloat16",
                storage_dtype_str="bfloat16",
            ),
            model_name="gpt2",
        ),
        trainsplit=SplitConfig(start=0, end=50, tokens_from_split=200_000),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**17,
            meta_batch_size=2**19,
            llm_batch_size=2**16,
        ),
        seq_len=256,
    )


def gpt_2_block(layer: int | list[int] | tuple[int], io="input"):
    if isinstance(layer, list | tuple):
        return gpt_2([f"{l}.{io}" for l in layer])
    return gpt_2(f"{layer}.{io}")


cfg = RunConfig[DynamicThreshConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block([6, 7]),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=256,
        optim="Adam",
        lr=1e-3,
        betas=(0.9, 0.997),
        #
        use_autocast=True,
        use_lars=True,
        #
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 0,
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=5000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=768 * 2, dict_mult=128),
    arch_cfg=DynamicThreshConfig(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=0.1,
            momentum=0.0,
            l0_diff_mult=1,
            lr=0.03,
            warmup_len=0,
            initial_value=1,
        ),  # Swept(1.5, 2.0, 1.0, 0.7, 3.0),
        l1_end_scale=0,  # Swept(0.0, 0.01),
    ),
)

arch = DynamicThreshSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")

sweep_manager.initialize_sweep()

sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
