from saeco_research.architectures.sweep_tg.orig_rewrite.tg_arch import (
    TGArch,
    TGSAEConfig,
)
from saeco_research.architectures.sweep_tg.orig_rewrite.threshgrad_v1 import (
    GateFunctionConfig,
    GatingConfig,
    ThreshGateConfig,
    ThreshGateConfigThing,
)

from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import (
    gpt_2_block,
)
from saeco.initializer import InitConfig
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

cfg = RunConfig[TGSAEConfig](
    train_cfg=TrainConfig(
        data_cfg=gpt_2_block(7),
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=50_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=2048 * 2,
        optim="Adam",
        lr=3e-4,
        betas=(0.9, 0.99),
        #
        use_autocast=False,
        use_lars=True,
        #
        weight_decay=1e-3,
        l0_target=50,
        l0_target_adjustment_size=0.0001,
        coeffs={
            "sparsity_loss": 1.5e-3,
            "L2_loss": 1,
        },
        #
        intermittent_metric_freq=50000,
        use_averaged_model=False,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=TGSAEConfig(
        decay_l1_steps=None,
        penalize_in_gate=True,
        penalize_after_gate=False,
        thresh_gate_cfg=ThreshGateConfig(
            deep_preprocessing=None,
            mag_weights=False,
            leniency_targeting=0.0,
            initial_leniency=0.73,
            signed_mag=False,
            gate_cfg=GatingConfig(
                gate_cfg=GateFunctionConfig(
                    uniform_noise=True,  # yeah kinda unsure on any of these
                    noise_mult=0.0,  # wait this is kinda very high right?
                    # no wait. it's maybe not so high. loss is MSE not L2 norm
                    # and normalized so each dim is about 1
                    # so.. yeah this is prob in the reasonable oom range
                    exp_mag=True,
                    p_off_noise=0.0,  # def not sure that's best on this one
                    gate_noise=False,  # !!
                    # this was on and it's doing well. trying w it off
                    # oh wow its garbage with it off wow lol
                    # it being off is like... clearly incorrect or something
                    # or maybe what's stupid is having p_off_noise not be zero
                    # and this basically sets it to zero but does extra stuff too
                    # where the extra stuff seems kinda bad. so I'm trying w it off and
                    # w p_off_noise=0.0
                    # the extra stuff being you get different grad sizes
                ),
                backward_cfg=ThreshGateConfigThing(
                    offgrad_mask_only_on_positive_noise=False,
                    out1_has_off_adjustment=True,
                    out1_has_on_adjustment=False,
                    fixed_off_leniency=None,
                    modify_grad_output=False,
                    out0_like_off_adj=True,
                ),
            ),
        ),
    ),
)
arch = TGArch(cfg)
sweep_manager = arch.get_sweep_manager()
sweep_manager.rand_run_no_agent(project="default-project")


# sweep_manager.initialize_sweep()
# sweep_manager.rand_run_no_agent()
# sweep_manager.run_manual_sweep_with_monitoring(new_pods=10)
