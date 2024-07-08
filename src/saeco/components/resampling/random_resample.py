from .resampler import Resampler
import torch


class RandomResampler(Resampler):
    def get_reset_feature_directions(self, num_directions, data_source, model):
        data = next(data_source)
        directions = torch.randn(num_directions, data.shape[1]).cuda().float()
        directions /= directions.norm(dim=1, keepdim=True)
        # directions /=
        self.bias_reset_value = -1e-1
        return directions
