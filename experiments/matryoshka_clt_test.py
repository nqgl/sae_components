from pathlib import Path

from saeco.architectures.matryoshka_clt import MatryoshkaCLT, MatryoshkaCLTConfig
from saeco.components.resampling.anthropic_resampling import (
    AnthResamplerConfig,
    OptimResetValuesConfig,
)
from saeco.data.data_cfg import DataConfig
from saeco.data.data_config_definitions import gemma_2_2b_openwebtext_bf16, gpt_2_block
from saeco.data.generation_config import DataGenerationProcessConfig
from saeco.data.model_cfg import ActsDataConfig, ModelConfig
from saeco.data.split_config import SplitConfig
from saeco.initializer import InitConfig
from saeco.initializer.initializer_config import InitConfig
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.trainer import RunSchedulingConfig
from saeco.trainer.run_config import RunConfig
from saeco.trainer.train_config import TrainConfig

PROJECT = "matryoshka_clt"

from saeco.mlog import mlog

mlog.init()

dict_mult = 8
n_sites = 12

input_mlp_sites = ["transformer.h.{}.mlp.input".format(i) for i in range(n_sites)]
output_mlp_sites = ["transformer.h.{}.mlp.output".format(i) for i in range(n_sites)]
mlp_sites = input_mlp_sites + output_mlp_sites

data_config = DataConfig(
    dataset="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
    model_cfg=ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            sites=mlp_sites,
            d_data=768,
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="bfloat16",
            storage_dtype_str="bfloat16",
        ),
        model_name="gpt2",
    ),
    trainsplit=SplitConfig(start=0, end=50, tokens_from_split=5_000_000),
    generation_config=DataGenerationProcessConfig(
        # tokens_per_pile=2**25,
        acts_per_pile=2**15,
        meta_batch_size=2**17,
        llm_batch_size=2**13,
        compress_acts=True,
    ),
    seq_len=256,
)


cfg = RunConfig[MatryoshkaCLTConfig](
    train_cfg=TrainConfig(
        data_cfg=data_config,
        raw_schedule_cfg=RunSchedulingConfig(
            run_length=100,
            resample_period=10_000,
            lr_warmup_length=2_000,
        ),
        #
        batch_size=512,
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
        input_sites=input_mlp_sites,
        target_sites=output_mlp_sites,
        intermittent_metric_freq=1000,
        use_averaged_model=False,
    ),
    resampler_config=AnthResamplerConfig(
        optim_reset_cfg=OptimResetValuesConfig(),
        expected_biases=2,
    ),
    #
    init_cfg=InitConfig(d_data=768 * n_sites, dict_mult=dict_mult),
    arch_cfg=MatryoshkaCLTConfig(n_sites=n_sites, n_nestings=3),
)

arch = MatryoshkaCLT(cfg)
arch.run_training()

arch.save_to_path(Path("matryoshka_clt_test_model.pt"))
