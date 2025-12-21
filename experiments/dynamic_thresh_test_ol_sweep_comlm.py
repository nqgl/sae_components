from comlm.storage import ComposerModelName

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
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.data.config.split_config import SplitConfig
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import EarlyStoppingBounds, TrainConfig


def s(x, *a):
    return x
    return Swept(x / 10, x / 3, x, x * 3, x * 10, *a)


model = ComposerModelName.from_str("1762986288-acoustic-asp")
data_cfg = DataConfig[ComlmModelConfig](
    override_dictpiler_path_str="/home/g/markov/sample_data_comlm",
    dataset="custom",
    model_cfg=ModelConfig[ComlmModelConfig](
        model_load_cfg=ComlmModelConfig(
            chk_ident=model.get_latest_downloaded_checkpoint()
        ),
        acts_cfg=ActsDataConfig(
            filter_pad=False,
            excl_first=False,
            d_data=512,
            sites=["layers.6.output.0"],  # .0 unpacks the tuple of (output, kv cache)
            storage_dtype_str="float32",
            autocast_dtype_str=None,
        ),
        torch_dtype_str="bfloat16",
    ),
    trainsplit=SplitConfig(start=0, end=80, tokens_from_split=None),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)


cfg = RunConfig[DynamicThreshConfig](
    train_cfg=TrainConfig(
        save_on_complete=True,
        data_cfg=data_cfg,
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=4_500 * 4,
            resample_period=Swept(300, 1_000, 3_000, 10_000, 100_000),
            lr_warmup_length=600,
        ),
        #
        batch_size=4096,
        optim=Swept("ScheduleFree"),
        lr=s(2e-3),
        weight_decay=Swept[float | None](None),
        betas=(0.9, 0.95),  # Swept((0.8, 0.9), (0.9, 0.9), (0.9, 0.95)),
        #
        use_autocast=False,
        use_lars=Swept(False),
        #
        l0_target=50,
        l0_target_adjustment_size=Swept(0.0003),
        coeffs={
            "sparsity_loss": Swept(0),
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=100_000_000,
        early_stopping_bounds=EarlyStoppingBounds(
            min_values={
                "L0": {
                    10: 1.0,
                    100: 1.0,
                    1000: 5.0,
                    1500: 10.0,
                    2000: 15.0,
                    2500: 15.0,
                    3000: 20.0,
                    4000: 30.0,
                }
            },
            max_values={
                "L2_loss": {
                    100: 3.0,
                    500: 1.0,
                    1000: 0.35,
                    2000: 0.3,
                }
                | dict.fromkeys(range(0, 4000, 100), 3.0),
                "L0": {
                    1000: 500.0,
                    4000: 70.0,
                }
                | {1000 + 100 * i: 120.01 + 40 * 8 - 8 * i for i in range(40)},
            },
        ),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
        bias_reset_value=-3.0,
        # freq_balance=25,
        dead_threshold=1e-6,
    ),
    #
    init_cfg=InitConfig(d_data=512, dict_mult=128),
    arch_cfg=DynamicThreshConfig(
        thresh_cfg=ThreshConfig(
            decay_toward_mean=s(0.0, 0.01, 0.03, 0.1, 0.3),
            momentum=Swept(0.5),
            l0_diff_mult=Swept(3, 10, 30, 100),
            lr=s(0.3, 0.1, 1.0),
            warmup_len=400,
            initial_value=2.0,
            stepclamp=s(1e-1),
            log_diff=False,
            # warmup_len=s(100),
        ),  # Swept(1.5, 2.0, 1.0, 0.7, 3.0),
        l1_end_scale=0.1,
        l1_decay_end=3_000,
        l1_decay_start=1_00,
    ),
)
cfg.to_swept_nodes().swept_combinations_count_including_vars()
arch = DynamicThreshSAE(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="nqgl/default-project")

# sweep_manager.initialize_sweep()

# sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
