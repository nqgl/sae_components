from functools import cached_property

from nnsight import LanguageModel, NNsight
from pydantic import Field
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from saeco.data.config.locations import DATA_DIRS

from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig


class HuggingFaceModelConfig(SweepableConfig):
    model_name: str = "gpt2"
    model_kwargs: dict = Field(default_factory=dict)
    _device: str = "cuda"
    # no_processing: bool = False

    def model_post_init(self, __context) -> None:
        assert not any(
            [
                v in self.model_kwargs
                for v in [
                    "trace",
                    "invoker_args",
                    "backend",
                    "remote",
                    "blocking",
                    "scan",
                ]
            ]
        ), "config's kwargs clash with nnsight::trace kwargs"

        assert len(self.acts_cfg.sites) == len(set(self.acts_cfg.sites))

        self._raw_model = None

        return super().model_post_init(__context)

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=DATA_DIRS.CACHE_DIR,
        )

    def _make_raw_model(self, load_as_dtype: torch.dtype | None = None):
        get_model_fn = (
            MODEL_FN_CALLABLE_OVERRIDE or AutoModelForCausalLM.from_pretrained
        )
        if self.torch_dtype_str is None:
            model = get_model_fn(
                self.model_name,
                cache_dir=DATA_DIRS.CACHE_DIR,
                device_map=self._device,
            )
        else:
            model = get_model_fn(
                self.model_name,
                cache_dir=DATA_DIRS.CACHE_DIR,
                torch_dtype=self.torch_dtype,
                device_map=self._device,
            )
        return model
        # if MODEL_FN_CALLABLE_OVERRIDE is not None:
        #     model = NNsight(model)

    def nnsight_wrap(self, model) -> NNsight:
        model = NNsight(model)
        if not hasattr(model, "tokenizer"):
            setattr(model, "tokenizer", self.tokenizer)


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
