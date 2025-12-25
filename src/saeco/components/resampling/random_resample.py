import torch

from .resampler import Resampler


class RandomResampler(Resampler):
    @torch.no_grad()
    def get_reset_feature_directions(self, num_directions, data_source, model):
        data = next(data_source)
        directions = torch.randn(num_directions, data.shape[1]).cuda().float()
        directions /= directions.norm(dim=1, keepdim=True)
        directions_re = model(directions * (768**0.5))
        directions_re /= directions_re.norm(dim=1, keepdim=True)
        error = directions - directions_re * (directions * directions_re).sum(
            dim=1, keepdim=True
        )
        directions = error
        directions /= directions.norm(dim=1, keepdim=True)
        # directions /=
        self.bias_reset_value = -3 * torch.randn(num_directions).cuda().float()
        return directions * 0.01


class RandomResampler(Resampler):
    @torch.no_grad()
    def get_reset_feature_directions(self, num_directions, data_source, model):
        data = next(data_source)
        directions = (
            torch.randn(max(2**10, num_directions), data.shape[1]).cuda().float()
        )
        directions /= directions.norm(dim=1, keepdim=True)
        directions *= 768**0.5
        directions_re = model(directions)
        error = directions - directions_re
        sqerr = (error**2).sum(dim=1)
        v, i = torch.topk(sqerr, k=num_directions)

        directions = directions[i]
        directions_re = directions_re[i]

        directions_re /= directions_re.norm(dim=1, keepdim=True)
        directions /= directions.norm(dim=1, keepdim=True)

        error = directions - directions_re * (directions * directions_re).sum(
            dim=1, keepdim=True
        )
        directions = error
        directions /= directions.norm(dim=1, keepdim=True)
        # directions /=
        self.bias_reset_value = -1e-1
        return directions
