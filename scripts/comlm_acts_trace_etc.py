# %%
from saeco.mlog import mlog

mlog.init(project="markov-bio/evaluator")


# %%
import torch
from comlm.utils import (
    ComposerModelName,
)

from saeco.data.config.data_cfg import (
    DataConfig,
    DataGenerationProcessConfig,
    SplitConfig,
)
from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.data.dict_batch.dict_batch import DictBatch
from saeco.misc.nnsite import getsite

model = "1761861357-rustling-mule"
# checkpoint = "ep0-ba198000-rank0.pt"
model = ComposerModelName.from_str("1762986288-acoustic-asp")
# model.get_latest_checkpoint_from_s3().fetch_from_s3()
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
tokens_data = data_cfg.tokens_data(split=data_cfg.trainsplit)
input_data = tokens_data.get_tokens(num_tokens=None)

meta_batch_size = data_cfg.generation_config.meta_batch_size // tokens_data.seq_len
input_data_split = input_data.split(meta_batch_size)

# %%
acts_data_creator = data_cfg.acts_data_creator()
a_data = input_data_split[0][:1]
assert isinstance(a_data, DictBatch)

# acts_data_creator.to_acts(
#     a_data, llm_batch_size=data_cfg.generation_config.llm_batch_size
# )
# %%
tx_data = data_cfg.model_cfg.model_load_cfg.input_data_transform(a_data)
d = {}
# %%
with acts_data_creator.cfg.model_cfg.autocast_context():
    with torch.inference_mode():  # is this ok with nnsight?
        model = acts_data_creator.model
        args, kwargs = (
            acts_data_creator.cfg.model_cfg.model_load_cfg.unpack_model_inputs(
                tx_data,
                extra_kwargs=acts_data_creator.cfg.model_cfg.model_kwargs,
            )
        )
        args = [kwargs.pop("tokens")]

        with model.trace(*args, **kwargs):
            for site in acts_data_creator.cfg.model_cfg.acts_cfg.sites:
                acts_module = getsite(model, site)
                acts = acts_module.save()
                d[site] = acts

# %%
# with acts_data_creator.cfg.model_cfg.autocast_context():
# with torch.inference_mode():  # is this ok with nnsight?
model = acts_data_creator.model
args, kwargs = acts_data_creator.cfg.model_cfg.model_load_cfg.unpack_model_inputs(
    tx_data,
    extra_kwargs=acts_data_creator.cfg.model_cfg.model_kwargs,
)
args = [kwargs.pop("tokens")]
with model.trace(*args, **kwargs) as m:
    # for site in acts_data_creator.cfg.model_cfg.acts_cfg.sites:
    acts_module = model.layers[6].output[0]
    acts = acts_module.save()  # d[site] = acts

# %%
acts
# %%
model = acts_data_creator.model
args, kwargs = acts_data_creator.cfg.model_cfg.model_load_cfg.unpack_model_inputs(
    tx_data,
    extra_kwargs=acts_data_creator.cfg.model_cfg.model_kwargs,
)
args = [
    kwargs.pop("tokens"),
    kwargs.pop("attention_mask"),
    kwargs.pop("ranks"),
    kwargs.pop("metadata"),
    kwargs.pop("frame_idx"),
]
with model.trace(*args, **kwargs) as m:
    # for site in acts_data_creator.cfg.model_cfg.acts_cfg.sites:
    acts_module = model.layers[6].output[0]
    acts2 = acts_module.save()  # d[site] = acts
# %%

(acts2 - acts).abs().max()
# %%
