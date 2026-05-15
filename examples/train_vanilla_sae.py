"""
Train a Vanilla SAE on GPT-2 activations
========================================

End-to-end training script. Wires together a `RunConfig` (with train,
resampler, init, and arch sub-configs) and runs a single training run.

The `arch_cfg.pre_bias` and `train_cfg.lr` fields below use `Swept(...)`
to declare a small grid (2 * 3 = 6 combinations). The script as written
runs ONE randomly-selected configuration via `rand_run_no_agent`. To run
the full grid (locally or on remote pods), see the commented-out
`sweep_manager` calls at the bottom.

Prerequisites
-------------
- A CUDA-capable GPU (`Architecture.__init__` defaults to `device="cuda"`).
- HuggingFace + transformers can fetch GPT-2 (~500 MB on first run).
- `pip install -e .` from the repo root.
- For W&B logging: `pip install -e ".[wandb]"`.
- For remote pod orchestration: `pip install -e ".[remote]"`.
"""

from saeco import (
    InitConfig,
    RunConfig,
    RunSchedulingConfig,
    Swept,
    TrainConfig,
)
from saeco.architectures.vanilla import VanillaConfig, VanillaSAE
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import gpt_2_block

cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        # Train on the input to GPT-2's block 6.
        data_cfg=gpt_2_block(layer=6),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        # Wrapping a value in Swept(...) makes that field one axis of a grid.
        lr=Swept(1e-3, 3e-4, 1e-4),
        batch_size=4096,
        optim="Adam",
        betas=(0.9, 0.997),
        use_autocast=True,
        use_lars=True,
        # L0 targeting nudges the sparsity penalty coefficient to hit a
        # target average L0 (number of active features per token).
        l0_target=50,
        l0_target_adjustment_size=0.001,
        coeffs={
            "sparsity_loss": 1.1e-3,
            "L2_loss": 1.0,
        },
        intermittent_metric_freq=1000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    init_cfg=InitConfig(d_data=768, dict_mult=32),
    arch_cfg=VanillaConfig(
        pre_bias=Swept(True, False),  # second axis of the grid
    ),
)


if __name__ == "__main__":
    arch = VanillaSAE(cfg)

    # Run a single randomly-selected configuration locally.
    sweep_manager = arch.get_sweep_manager()
    sweep_manager.rand_run_no_agent()

    # To run the full grid, uncomment one of:
    #
    #   # Initialize a sweep that other workers can pull configs from:
    #   sweep_manager.initialize_sweep()
    #
    #   # Spin up 6 remote pods (requires `pip install -e ".[remote]"` and
    #   # ezpod credentials) and run the full grid across them with
    #   # monitoring:
    #   sweep_manager.run_manual_sweep_with_monitoring(new_pods=6)
