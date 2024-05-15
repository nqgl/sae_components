from dataclasses import dataclass

from nqgl.mlutils.components.component_layer.resampler.resampler import ResamplingMethod
import torch


@dataclass
class TopKResamplingConfig:
    resample_top_k: int = None


class TopKResampling(ResamplingMethod):
    def get_directions(self, cache, x, y_pred, y):
        indices = self.get_topk_indices(cache, x, y_pred, y)
        return self.get_directions_for_indices(cache, x, y_pred, y, indices)

    # @abstractmethod
    def get_topk_indices(self, cache, x, y_pred, y):
        ranking = self.get_ranking_metric(cache, x, y_pred, y)
        k = self.cfg.resample_top_k or self.cfg.num_to_resample
        return torch.topk(ranking, k, largest=True).indices

    def get_ranking_metric(self, cache, x, y_pred, y):
        return (y - y_pred).pow(2).mean(dim=-1)

    def get_directions_for_indices(self, cache, x, y_pred, y, indices):
        x = x[indices]
        y_pred = y_pred[indices]
        y = y[indices]
        return super().get_directions(cache, x, y_pred, y)
