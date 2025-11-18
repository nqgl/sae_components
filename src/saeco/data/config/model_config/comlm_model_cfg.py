from functools import cached_property
from typing import Any, cast

import torch
from comlm.architecture import TransformerArchitecture
from comlm.datasource.training_batch import NoisedBatch
from comlm.exprank import XRArch
from comlm.exprank.xr_transformer import XRTransformer
from comlm.utils import ModelCheckpointIdentifier
from nnsight import NNsight
from paramsight import get_resolved_typevars_for_base, takes_alias

from saeco.data.config.model_config.model_type_cfg_base import ModelLoadingConfigBase
from saeco.data.dict_batch.dict_batch import DictBatch


class ComlmModelConfig[ArchT: TransformerArchitecture[Any, Any, Any] = XRArch](
    ModelLoadingConfigBase[XRTransformer]
):
    chk_ident: ModelCheckpointIdentifier

    @takes_alias
    @classmethod
    def get_arch_cls(cls) -> type[ArchT]:
        return cast(
            type[ArchT], get_resolved_typevars_for_base(cls, ComlmModelConfig)[0]
        )

    @cached_property
    def pretrained_arch(self) -> ArchT:
        arch_cls = self.get_arch_cls()
        print(f"loading {arch_cls} from {self.chk_ident.path}")
        return arch_cls.load_from_checkpoint_identifier(
            self.chk_ident,
            load_weights_only=True,
            update_arch_before_load=self.update_arch_before_load,
            update_arch_after_load=self.update_arch_after_load,
            # dont_load_checkpoint=self.run_cfg.finetuner_cfg.skip_loading_weights,#TODO
            strict_model_weights=True,
            state_dict_ignore_keys=self.state_dict_modify,
            skip_mlog_init=True,
            project_name=None,
        )

    @property
    def name(self) -> str:  # type: ignore
        return self.chk_ident.simple_name

    @cached_property
    def tokenizer(self):
        return self.pretrained_arch.tokenizer

    def _make_raw_model(
        self,
        load_as_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ):
        self.pretrained_arch.composer_model.compile_model = False
        self.pretrained_arch.composer_model.eval()
        return self.pretrained_arch.composer_model

    def nnsight_wrap(self, model) -> NNsight:
        return NNsight(model)

    def update_arch_before_load(self, arch: TransformerArchitecture):
        # disable any preemptive data loading
        arch.run_cfg.train_cfg.evals_cfg.spearman_min_gene_counts = []
        arch.run_cfg.train_cfg.evals_cfg.ce_min_gene_counts = []
        arch.run_cfg.train_cfg.checkpoint_interval_batches = None

    def update_arch_after_load(self, arch: TransformerArchitecture):
        pass

    def state_dict_modify(self, state_dict: dict) -> None:
        ...
        # model_dict = state_dict["state"]["model"]
        # keysl = list(model_dict.keys())
        # for key in keysl:
        #     for rmname in self.STATE_DICT_REMOVE_KEYS:
        #         if rmname in key:
        #             del model_dict[key]

    def input_data_transform(self, input_data: DictBatch) -> DictBatch:
        return NoisedBatch.construct_with_other_data(input_data).cuda()

    def filter_acts(self, input_data: DictBatch, acts: DictBatch) -> DictBatch:
        return acts


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
