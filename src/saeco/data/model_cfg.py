from functools import cached_property

from nnsight import LanguageModel, NNsight
from pydantic import Field

from transformers import AutoModelForCausalLM, AutoTokenizer

from saeco.data.locations import DATA_DIRS

from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig

MODEL_FN_CALLABLE_OVERRIDE = None


class ActsDataConfig(SweepableConfig):
    d_data: int = 768
    sites: list[str] = ["transformer.h.6.input"]
    excl_first: bool = True
    filter_pad: bool = True
    storage_dtype_str: str | None = None
    autocast_dtype_str: str | bool | None = None
    force_cast_dtype_str: str | None = None

    @property
    def actstring(self):
        sites_str = "_".join(sorted(self.sites))
        return f"{sites_str}_{self.excl_first}_{self.filter_pad}_{self.storage_dtype_str}_{self.autocast_dtype_str}_{self.force_cast_dtype_str}"

    @property
    def storage_dtype(self):
        if self.storage_dtype_str is None:
            return None
        return str_to_dtype(self.storage_dtype_str)

    @property
    def autocast_dtype(self):
        if self.autocast_dtype_str is False:
            return False
        if self.autocast_dtype_str is None:
            return self.storage_dtype
        return str_to_dtype(self.autocast_dtype_str)

    @property
    def force_cast_dtype(self):
        if self.force_cast_dtype_str is None:
            return None
        return str_to_dtype(self.force_cast_dtype_str)


class ModelConfig(SweepableConfig):
    model_name: str = "gpt2"
    acts_cfg: ActsDataConfig = Field(default_factory=ActsDataConfig)
    model_kwargs: dict = Field(default_factory=dict)
    _device: str = "cuda"
    # no_processing: bool = False
    torch_dtype_str: str | None = None

    @property
    def torch_dtype(self):
        if self.torch_dtype_str is None:
            return None
        return str_to_dtype(self.torch_dtype_str)

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

    def _make_raw_model(self):
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

    @property
    def raw_model(self) -> AutoModelForCausalLM:
        if self._raw_model is None:
            self._raw_model = self._make_raw_model()
        return self._raw_model

    @property
    def model(self) -> NNsight:
        model = NNsight(self.raw_model)
        if not hasattr(model, "tokenizer"):
            setattr(model, "tokenizer", self.tokenizer)
        return model

    @model.setter
    def model(self, m: NNsight):
        self._raw_model = m

    @property
    def modelstring(self) -> str:
        return f"{self.model_name}_{self.torch_dtype_str}_{self.acts_cfg.actstring}"


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
