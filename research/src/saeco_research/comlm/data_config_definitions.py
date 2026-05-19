from comlm.config.configs import ComposerRunConfig
from comlm.datasource.data_config_definitions import tahoe_data_config
from comlm.exprank.XRTransformerConfig import XRTransformerConfig
from comlm.storage import ComposerModelName

from saeco.architecture.arch_reload_info import ArchRef
from saeco.data.config.data_cfg import DataConfig
from saeco.data.config.generation_config import DataGenerationProcessConfig
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.data.config.split_config import SplitConfig
from saeco_research.comlm.comlm_model_cfg import ComlmModelConfig

saeco_tahoe_data_cfg = DataConfig[ComlmModelConfig](
    override_token_dictpiler_path_str="/home/g/workspace/tahoe_batches",
    dataset="tahoe_bulked",
    model_cfg=ModelConfig[ComlmModelConfig](
        model_load_cfg=ComlmModelConfig(
            chk_ident=ComposerModelName.from_str(
                "1762986288-acoustic-asp"
            ).get_latest_downloaded_checkpoint(),
            inject_arch_data_cfg=tahoe_data_config,
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
    trainsplit=SplitConfig(start=0, end=100, tokens_from_split=None),
    generation_config=DataGenerationProcessConfig(
        acts_per_pile=2**18,
        meta_batch_size=2**18,
        llm_batch_size=2**13,
    ),
    seq_len=1024,
)
saeco_tahoe_data_cfg.model_cfg.model_load_cfg.name

model = ComposerModelName.from_str("1762986288-acoustic-asp")

model = ComposerModelName.from_str("1767399915-fragrant-cuttlefish")


comlm_512_orig = ComposerModelName.from_str("1762986288-acoustic-asp")
comlm_768_nodrop = ComposerModelName.from_str("1767399915-fragrant-cuttlefish")
comlm_768_dropout = ComposerModelName.from_str("1767413438-impartial-owl")
comlm_768_nodrop_noperm = ComposerModelName.from_str("1767385398-ninja-hummingbird")


def get_data_cfg(
    model: ComposerModelName,
    sites: list[str] | tuple[str, ...] = ("layers.6.output.0",),
    d_data: int | None = None,
    seq_len: int = 2048,
) -> DataConfig[ComlmModelConfig]:
    arch_ref = ArchRef[ComposerRunConfig[XRTransformerConfig]].model_validate_json(
        model.arch_ref_path.read_text()
    )
    return DataConfig[ComlmModelConfig](
        override_token_dictpiler_path_str="/home/g/workspace/data/sample_data_comlm_larger",
        dataset="custom",
        model_cfg=ModelConfig[ComlmModelConfig](
            model_load_cfg=ComlmModelConfig(
                chk_ident=model.get_latest_downloaded_checkpoint()
            ),
            acts_cfg=ActsDataConfig(
                filter_pad=False,
                excl_first=False,
                d_data=d_data or arch_ref.config.arch_cfg.d_model * len(sites),
                sites=list(sites),
                storage_dtype_str="bfloat16",
                autocast_dtype_str="bfloat16",
            ),
            torch_dtype_str="float32",
        ),
        trainsplit=SplitConfig(start=0, end=80, tokens_from_split=None),
        generation_config=DataGenerationProcessConfig(
            # tokens_per_pile=2**25,
            acts_per_pile=2**18,
            meta_batch_size=2**18,
            llm_batch_size=2**16,
        ),
        seq_len=seq_len,
    )


if __name__ == "__main__":
    from saeco.mlog import mlog

    mlog.init(project="markov-bio/evaluator")
    cfg = get_data_cfg(comlm_768_nodrop)
    cfg.store_split(cfg.trainsplit)
    # saeco_tahoe_data_cfg.store_split(saeco_tahoe_data_cfg.trainsplit)


def convert_to_tahoe(
    cfg: DataConfig[ComlmModelConfig], **kwargs
) -> DataConfig[ComlmModelConfig]:
    return cfg.model_copy(
        update={
            **dict(
                override_token_dictpiler_path_str="/home/g/workspace/tahoe_batches_full",
                dataset="tahoe_bulked_full",
                trainsplit=SplitConfig(start=0, end=100, tokens_from_split=None),
                seq_len=2048,
                model_cfg=cfg.model_cfg.model_copy(
                    update={
                        "model_load_cfg": cfg.model_cfg.model_load_cfg.model_copy(
                            update={
                                "inject_arch_data_cfg": tahoe_data_config,
                            }
                        )
                    }
                ),
            ),
            **kwargs,
        }
    )


if __name__ == "__main__":
    from saeco.mlog import mlog

    mlog.init(project="markov-bio/evaluator")
    for model in [comlm_768_nodrop, comlm_768_dropout, comlm_768_nodrop_noperm]:
        cfg = get_data_cfg(model)
        cfg.store_split(cfg.trainsplit)
