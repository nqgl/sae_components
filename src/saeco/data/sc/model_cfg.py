from transformer_lens import HookedTransformer, utils


from dataclasses import dataclass, field


@dataclass
class ActsDataConfig:
    d_data: int = 768
    site: str = "resid_pre"
    layer_num: int = 6
    excl_first: bool = False

    @property
    def hook_site(self):
        return utils.get_act_name(self.site, self.layer_num)


@dataclass
class ModelConfig:
    model_name: str = "gpt2-small"
    acts_cfg: ActsDataConfig = field(default_factory=ActsDataConfig)
    # d_data: int = 768

    def __post_init__(self):
        model = None

        def getmodel():
            nonlocal model
            if model is None:
                model = HookedTransformer.from_pretrained(self.model_name)
            return model

        def setmodel(m):
            nonlocal model
            assert model is None, "model already set"
            model = m

        self._setmodel = setmodel
        self._getmodel = getmodel

    @property
    def model(self) -> HookedTransformer:
        return self._getmodel()

    @model.setter
    def model(self, m: HookedTransformer):
        self._setmodel(m)
