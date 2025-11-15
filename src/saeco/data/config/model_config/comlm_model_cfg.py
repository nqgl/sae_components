from functools import cached_property
from typing import cast

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from saeco.data.config.locations import DATA_DIRS
from typing import Any
from saeco.data.config.model_config.model_type_cfg_base import ModelLoadingConfigBase
from comlm.utils import ModelCheckpointIdentifier
from comlm.architecture import TransformerArchitecture
from comlm.exprank import XRArch
from comlm.exprank.xr_transformer import XRTransformer
from paramsight import takes_alias, get_resolved_typevars_for_base
from nnsight import NNsight

from saeco.data.dict_batch.dict_batch import DictBatch


class ComlmModelConfig[ArchT: TransformerArchitecture[Any, Any, Any] = XRArch](
    ModelLoadingConfigBase[XRTransformer]
):
    model_ident: ModelCheckpointIdentifier

    @takes_alias
    @classmethod
    def get_arch_cls(cls) -> type[ArchT]:
        return cast(
            type[ArchT], get_resolved_typevars_for_base(cls, ComlmModelConfig)[0]
        )

    @cached_property
    def pretrained_arch(self) -> ArchT:
        arch_cls = self.get_arch_cls()
        print(f"loading {arch_cls} from {self.model_ident.path}")
        return arch_cls.load_from_checkpoint_identifier(
            self.model_ident,
            load_weights_only=True,
            update_arch_before_load=self.update_arch_before_load,
            update_arch_after_load=self.update_arch_after_load,
            # dont_load_checkpoint=self.run_cfg.finetuner_cfg.skip_loading_weights,#TODO
            strict_model_weights=True,
            state_dict_ignore_keys=self.state_dict_modify,
            skip_mlog_init=True,
        )

    @property
    def model_name(self) -> str:  # type: ignore
        return self.model_ident.simple_name

    @cached_property
    def tokenizer(self):
        return self.pretrained_arch.tokenizer

    def _make_raw_model(
        self,
        load_as_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ) -> XRTransformer:
        return self.pretrained_arch.model

    def nnsight_wrap(self, model: XRTransformer) -> NNsight:
        return NNsight(model)

    def update_arch_before_load(self, arch: TransformerArchitecture):
        pass

    def update_arch_after_load(self, arch: TransformerArchitecture):
        pass

    def state_dict_modify(self, state_dict: dict):
        # model_dict = state_dict["state"]["model"]
        # keysl = list(model_dict.keys())
        # for key in keysl:
        #     for rmname in self.STATE_DICT_REMOVE_KEYS:
        #         if rmname in key:
        #             del model_dict[key]
        return state_dict


    def input_data_transform(self, input_data: DictBatch) -> DictBatch:
        
# train_dataloader_filtered
# [
#     train_datagen_unfiltered[
#         self.run_cfg.train_cfg.data_cfg.get_dataset(
#                 batch_size=self.device_batch_size,
#                 num_workers=num_workers,
#             )
#         ->
#         self._add_metadata_processing_to_dataset(
#             cast(Iterable[dict[str, torch.Tensor]], dataset)
#         )
#     ]
#      ->
#      _convert_to_filtered_dataloader

# ]
# ->
# self.training_microbatch_transform
