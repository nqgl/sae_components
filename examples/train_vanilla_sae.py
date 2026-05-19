"""
Train a Vanilla SAE on Gemma-4 (instruction-tuned) activations
==============================================================

End-to-end training script. Wires together a `RunConfig` (with train,
resampler, init, and arch sub-configs) and runs a single training run.

Trains on the input to layer 14 of the instruction-tuned Gemma-4 E2B
checkpoint, over user/assistant conversations from lmsys-chat-1m
(CONVERSATION tokenization, PAD-packed so turn boundaries aren't
scrambled across unrelated conversations).

The `arch_cfg.pre_bias` and `train_cfg.lr` fields below use `Swept(...)`
to declare a small grid (2 * 3 = 6 combinations). The script as written
runs ONE randomly-selected configuration via `rand_run_no_agent`. To run
the full grid (locally or on remote pods), see the commented-out
`sweep_manager` calls at the bottom.

Prerequisites
-------------
- A CUDA-capable GPU (`Architecture.__init__` defaults to `device="cuda"`).
- The Gemma-4 checkpoint is **gated** on HuggingFace: accept the license
  on the model page and authenticate (`huggingface-cli login` or export
  `HF_TOKEN`) before first run. The weights are a multi-GB download.
- `pip install -e ./sweepable && pip install -e .` from the repo root
  (W&B logging is included; it is a core dependency).
- For remote pod orchestration: `pip install -e ".[remote]"`.

Note: `GEMMA_4_DEFAULT_D_DATA` is the repo's residual-stream width
placeholder for the E2B variant — confirm it against the loaded model
if you change checkpoints.
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
from saeco.data.config.data_config_definitions import (
    GEMMA_4_DEFAULT_D_DATA,
    gemma_4_lmsys_chat,
)

cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        # Train on the input to layer 14 of instruction-tuned Gemma-4.
        data_cfg=gemma_4_lmsys_chat(14),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        # Wrapping a value in Swept(...) makes that field one axis of a grid.
        lr=Swept(1e-3, 3e-4, 1e-4),
        batch_size=4096,
        optim="Adam",
        betas=(0.9, 0.99),
        use_autocast=True,
        use_lars=False,
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
    init_cfg=InitConfig(d_data=GEMMA_4_DEFAULT_D_DATA, dict_mult=32),
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
