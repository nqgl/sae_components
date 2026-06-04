"""Smoke/training config for the cgraph-v2 Gated SAE on cached gemma-4 data.

Mirrors ``dynamic_thresh_gemma4.py`` (same data, init, l0 target, coeffs) but
runs ``GatedGraphSAE`` -- the gated arch rebuilt on the interned graph core. The
cached data for this config only covers ~750 steps, so the run stops when the
buffer is exhausted; that's enough to confirm the graph-native arch trains
through the real ``Trainer``.

``intermittent_metric_freq`` is set high to skip the subject-model reconstruction
eval (a separate integration surface) on this first run.

Run:
    SAECO ... CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
        PYTHONPATH=research/src python research/experiments/gated_graph_gemma4.py
"""

from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import gemma_4_lmsys_chat
from saeco.initializer import InitConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import EarlyStoppingBounds, TrainConfig
from saeco_research.architectures.gated_graph import GatedGraphConfig, GatedGraphSAE

cfg = RunConfig[GatedGraphConfig](
    train_cfg=TrainConfig(
        save_on_complete=False,
        data_cfg=gemma_4_lmsys_chat(14, num_train_tokens=1_000_000),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=4_500 * 4,
            resample_period=10_000,
            lr_warmup_length=600,
        ),
        batch_size=1024,
        optim="Adam",
        lr=2e-3,
        weight_decay=None,
        betas=(0.9, 0.95),
        use_autocast=False,
        use_lars=False,
        l0_target=50,
        l0_target_adjustment_size=0.0003,
        coeffs={
            "sparsity_loss": 3e-5,
            "L2_loss": 1,
        },
        intermittent_metric_freq=5_000,  # skip recons eval on this first run
        early_stopping_bounds=EarlyStoppingBounds(
            min_values={
                "L0": {10: 1.0, 100: 1.0, 1000: 5.0},
            },
            max_values={
                "L2_loss": {100: 3.0, 500: 1.0, 1000: 0.35},
                "L0": {1000: 500.0},
            },
        ),
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    init_cfg=InitConfig(d_data=1536, dict_mult=32),
    arch_cfg=GatedGraphConfig(pre_bias=False, detach=True),
)

if __name__ == "__main__":
    arch = GatedGraphSAE(cfg)
    sweep_manager = arch.get_sweep_manager()
    sweep_manager.rand_run_no_agent(project="gated-graph-test")
