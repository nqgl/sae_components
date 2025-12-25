from contextlib import nullcontext
from functools import cached_property
from typing import Any

import torch
from nnsight import NNsight
from pydantic import Field

from saeco.data.config.model_config.acts_data_cfg import ActsDataConfig
from saeco.data.config.model_config.hf_model_cfg import HuggingFaceModelConfig
from saeco.data.config.model_config.model_type_cfg_base import ModelLoadingConfigBase
from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig


class ModelConfig[ModelLoadT: ModelLoadingConfigBase[Any] = HuggingFaceModelConfig](
    SweepableConfig
):
    model_load_cfg: ModelLoadT
    acts_cfg: ActsDataConfig = Field(default_factory=ActsDataConfig)
    _device: str = "cuda"
    # no_processing: bool = False
    torch_dtype_str: str | None = None
    model_kwargs: dict = Field(default_factory=dict)
    positional_args: list[str] = Field(default_factory=list)

    @property
    def model_name(self):
        return self.model_load_cfg.name

    @property
    def torch_dtype(self):
        if self.torch_dtype_str is None:
            return None
        return str_to_dtype(self.torch_dtype_str)

    def model_post_init(self, __context) -> None:
        assert not any(
            v in self.model_kwargs
            for v in [
                "trace",
                "invoker_args",
                "backend",
                "remote",
                "blocking",
                "scan",
            ]
        ), "config's kwargs clash with nnsight::trace kwargs"

        assert len(self.acts_cfg.sites) == len(set(self.acts_cfg.sites))

        self._raw_model = None

        return super().model_post_init(__context)

    @cached_property
    def tokenizer(self):
        return self.model_load_cfg.tokenizer

    @property
    def raw_model(self):
        if self._raw_model is None:
            self._raw_model = self.model_load_cfg._make_raw_model(
                load_as_dtype=self.torch_dtype,
                device=self._device,
            )
        return self._raw_model

    @property  # I have a vague memory that theres a reason this isn't a cached property?
    def model(self) -> NNsight:
        return self.model_load_cfg.nnsight_wrap(self.raw_model)

    @model.setter
    def model(self, m: NNsight):
        self._raw_model = m

    @property
    def modelstring(self) -> str:
        return f"{self.model_name}_{self.torch_dtype_str}_{self.acts_cfg.actstring}"

    def autocast_context(self):
        if self.acts_cfg.autocast_dtype is False:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=(self.acts_cfg.autocast_dtype or self.torch_dtype),
        )


def main():
    acts_cfg = ActsDataConfig(
        excl_first=True,
        d_data=2304,
        sites=["model.layers.17.input", "model.layers.18.output"],
        storage_dtype_str="bfloat16",
        autocast_dtype_str=None,
    )

    print(acts_cfg.actstring)


if __name__ == "__main__":
    main()
