from pydantic import Field
from transformer_lens import HookedTransformer, utils

from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig


class ActsDataConfig(SweepableConfig):
    d_data: int = 768
    site: str = "resid_pre"
    layer_num: int = 6
    excl_first: bool = True
    filter_pad: bool = True

    @property
    def hook_site(self):
        return utils.get_act_name(self.site, self.layer_num)

    @property
    def actstring(self):
        return f"{self.site}_{self.layer_num}_{self.excl_first}"


class ModelConfig(SweepableConfig):
    model_name: str = "gpt2-small"
    acts_cfg: ActsDataConfig = Field(default_factory=ActsDataConfig)
    model_kwargs: dict = Field(default_factory=dict)
    _device: str = "cuda"
    no_processing: bool = False
    torch_dtype: str | None = None

    def model_post_init(self, __context: utils.Any) -> None:
        # super().__init__(**data)
        model = None

        def getmodel():
            nonlocal model
            if model is None:
                if self.no_processing:
                    gen_fn = HookedTransformer.from_pretrained_no_processing
                else:
                    gen_fn = HookedTransformer.from_pretrained
                if self.torch_dtype is None:
                    model = gen_fn(
                        self.model_name,
                        device=self._device,
                    )
                else:
                    model = gen_fn(
                        self.model_name,
                        torch_dtype=str_to_dtype(self.torch_dtype),
                        # device_map=,
                        device=self._device,
                    )
            return model

        def setmodel(m):
            nonlocal model
            assert model is None, "model already set"
            model = m

        self._setmodel = setmodel
        self._getmodel = getmodel
        return super().model_post_init(__context)

    @property
    def model(self) -> HookedTransformer:
        return self._getmodel()

    @model.setter
    def model(self, m: HookedTransformer):
        self._setmodel(m)

    @property
    def modelstring(self):
        return f"{self.model_name}_{self.acts_cfg.actstring}"
