from nnsight import LanguageModel, NNsight
from pydantic import Field

from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig

MODEL_FN_CALLABLE_OVERRIDE = None


class ActsDataConfig(SweepableConfig):
    d_data: int = 768
    site: str = "transformer.h.6.input"
    excl_first: bool = True
    filter_pad: bool = True

    @property
    def actstring(self):
        return f"{self.site}_{self.excl_first}"


class ModelConfig(SweepableConfig):
    model_name: str = "gpt2"
    acts_cfg: ActsDataConfig = Field(default_factory=ActsDataConfig)
    model_kwargs: dict = Field(default_factory=dict)
    _device: str = "cuda"
    no_processing: bool = False
    torch_dtype_str: str | None = None

    @property
    def torch_dtype(self):
        if self.torch_dtype_str is None:
            return None
        return str_to_dtype(self.torch_dtype_str)

    def model_post_init(self, __context) -> None:
        # super().__init__(**data)
        model = None
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

        def getmodel():
            nonlocal model
            if model is None:
                get_model_fn = MODEL_FN_CALLABLE_OVERRIDE or LanguageModel
                if self.torch_dtype_str is None:
                    model = get_model_fn(
                        self.model_name,
                        device_map=self._device,
                    )
                else:
                    model = get_model_fn(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        device_map=self._device,
                        # dtype=self.torch_dtype,
                    )
                if MODEL_FN_CALLABLE_OVERRIDE is not None:
                    model = NNsight(model)

            return model

        def setmodel(m):
            nonlocal model
            assert model is None, "model already set"
            model = m

        self._setmodel = setmodel
        self._getmodel = getmodel
        return super().model_post_init(__context)

    @property
    def model(self) -> LanguageModel:
        return self._getmodel()

    @model.setter
    def model(self, m: LanguageModel):
        self._setmodel(m)

    @property
    def modelstring(self) -> str:
        return f"{self.model_name}_{self.acts_cfg.actstring}"
