from saeco.architectures.vanilla import VanillaConfig, VanillaSAE
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.config.data_config_definitions import (
    gpt_2_block,
)
from saeco.initializer import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

dcfg = gpt_2_block()
# dcfg.databuffer_num_workers = Swept(0, 1, 2, 16, 64)
# dcfg.databuffer_queue_size = Swept(0, 2, 8, 32, 128, 512)
dcfg.databuffer_worker_queue_base_size = Swept(0, 1, 2, 4, 8, 16)
dcfg.databuffer_worker_offset_mult = Swept[None | int](None, 0, 1, 2, 4)
cfg = RunConfig[VanillaConfig](
    train_cfg=TrainConfig(
        data_cfg=dcfg,
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=50_000,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=4096,
        optim="Adam",
        # Wrapping the lr values in a Swept designates this
        # field to be swept across in the upcoming grid search
        lr=1e-3,
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
        intermittent_metric_freq=1000,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=1,
    ),
    #
    init_cfg=InitConfig(d_data=768, dict_mult=8),
    arch_cfg=VanillaConfig(
        # And here we sweep across the pre_bias config option
        pre_bias=False,
    ),
)

arch = VanillaSAE(cfg)
sweep_manager = arch.get_sweep_manager()
# sweep_manager.rand_run_no_agent()
with sweep_manager.created_pods(12, keep=True) as pods:
    # with sweep_manager.created_pods(60, keep=True) as pods:
    print(len(pods.pods))
    # pods.run_manual_challenge_file(
    res = pods.prune(6, "src/saeco/sweeps/challenge.py", stop_after_n_complete=35)
    for pod in pods:
        print(pod.data.name, pod.output.end_time)
        if pod.output.end_time:
            print(pod.data.name, pod.output.end_time - pod.output.start_time)
    res

    print()
with sweep_manager.existing_pods() as pods:
    # with sweep_manager.created_pods(60, keep=True) as pods:
    assert len(pods.pods) == 6
sweep_manager.initialize_sweep(project="speedtest")
sweep_manager.run_manual_sweep_with_monitoring(0, purge_after=False, keep_after=True)
print()
