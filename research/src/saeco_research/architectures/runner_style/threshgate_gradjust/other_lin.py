import torch
from torch.optim.optimizer import Optimizer as Optimizer

import saeco.components.features as ft
from saeco.misc import lazycall


class RandResampledFP(ft.FeaturesParam):
    @torch.no_grad()
    def resample(self, *, indices, new_directions, bias_reset_value, optim: Optimizer):
        avg_other_norm = 1
        if not indices.all():
            avg_other_norm = self.features[~indices].norm(dim=-1).mean()
        new_directions = torch.randn_like(self.features[indices])
        new_directions /= new_directions.norm(dim=-1, keepdim=True)
        new_directions *= avg_other_norm * 0.2
        self.features[indices] = new_directions
        if not self.reset_optim_on_resample:
            return
        optim_state = self.optimstate(optim)
        fields = set(optim_state.keys()) - set(self.field_handlers.skips)

        for field in fields:
            if self.get_optim(optim).state[field] == {}:
                continue
            field_state = optim_state[field]
            assert field_state[:].shape == self.features.shape, (
                f"{field}: {field_state[:].shape} != {self.features.shape}"
            )

            optim_state[field, indices] = self.field_handlers.get_value(
                field=field,
                param=self,
                ft_optim_field_state=optim_state[field],
                feat_mask=indices,
                new_directions=new_directions,
            )


class OtherLinear(ft.LinWeights):
    def __init__(self, lin, weight_param_index=0, channel_split=False):
        super().__init__(lin)
        self.weight_param_index = weight_param_index
        self.channel_split = channel_split

    @property
    @lazycall
    def features(self):
        return {
            "weight": RandResampledFP(
                self.get_weight(),
                self.weight_param_index,
                feature_parameter_type="other",
            ),
            **(
                {
                    "bias": ft.FeaturesParam(
                        self.get_bias(), 0, feature_parameter_type="bias"
                    )
                }
                if self.bias is not None
                else {}
            ),
        }
