from dataclasses import dataclass, field
from transformer_lens import HookedTransformer
from sae_components.trainer import Trainable
from sae_components.components.losses import L2Loss, SparsityPenaltyLoss


@dataclass
class SAEConfig:
    layer: int
    model_name: str
    site: str
    d_data: int = 768
    expansion_factor: int = 8

    def __post_init__(self):
        model = None

        def getmodel():
            nonlocal model
            if model is None:
                model = HookedTransformer.from_pretrained(self.model_name)
            return model

        self._getmodel = getmodel

    @property
    def model(self) -> HookedTransformer:
        return self._getmodel()


class SAE(Trainable):
    def __init__(self, cfg, models, losses=None):
        super().__init__(models, losses)
        losses = losses or [L2Loss(), SparsityPenaltyLoss()]
        self.cfg = cfg
