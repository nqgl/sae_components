from .resampler import Resampler
import torch


class RandomResampler(Resampler):
    def get_reset_feature_directions(self, num_directions, data_source):
        data = next(data_source)
        return torch.randn(num_directions, data.shape[1]).cuda().float()
