# %%
from load_comlm import root_eval

# %%
from saeco.data.config.model_config.comlm_model_cfg import ComlmModelConfig
from saeco.data.config.model_config.model_cfg import ModelConfig
from saeco.evaluation.evaluation import Evaluation
from comlm.datasource.training_batch import NoisedBatch
import torch
# %%

root: Evaluation[NoisedBatch] = root_eval
# %%
# ADD METADATA

model_cfg: ModelConfig[ComlmModelConfig] = root.sae_cfg.train_cfg.data_cfg.model_cfg  # type: ignore

arch = model_cfg.model_load_cfg.pretrained_arch
comlm_cfg = arch.run_cfg
# %%

arch.metadata_tokenizer
# %%
root.features[0].value.device
# %%
root.features[1].indices()
# %%
metadata_tokenizer = arch.metadata_tokenizer
for key in comlm_cfg.arch_cfg.metadata_embedding_config.selected_metadata:
    tokenizer = metadata_tokenizer.tokenizers[key]


# %%
"""
maybe we will want some "compare intervention on input data" method

Also, I should probably drop out either all or none of the metadata for 

"""


data0 = root.docs[0:4]
# %%

root.get_inputs_type()
# %%
root.saved_acts.get_inputs_type()
# %%
#
# Initialize metadata if necessary
#
metadata_keys = comlm_cfg.arch_cfg.metadata_embedding_config.selected_metadata
has_metadata = [key in root.metadatas for key in metadata_keys]
if not all(has_metadata):
    assert not any(has_metadata), "Some metadata keys are already present"
    all_metadata_builder = root.metadata_builder(
        dtype=torch.long,
        device="cpu",
        item_size=(len(metadata_keys),),
    )
    for chunk in all_metadata_builder:
        all_metadata_builder << chunk.tokens.value["metadata"]
    for i, key in enumerate(metadata_keys):
        tokenizer = metadata_tokenizer.tokenizers[key]
        metadata = all_metadata_builder.value[:, i]
        root.metadatas[key] = metadata
        root.metadatas.set_str_translator(
            key, {"<<PAD>>": 0, "<<UNK>>": 1, **tokenizer.tokens}
        )
# %%
root.top_activations_and_metadatas(
    7,
    k=4,
    metadata_keys=metadata_keys,
)

# %%
