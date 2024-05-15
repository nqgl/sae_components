from nqgl.mlutils.components.component_layer.resampler import (
    ResamplingMethod,
    ResamplerConfig,
)
from dataclasses import dataclass
from nqgl.mlutils.components.component_layer.resampler.methods.orthogonalize import (
    re_init_neurons_gram_shmidt_precise_iterative_argmax,
)
from nqgl.mlutils.components.component_layer.resampler.methods.topk import (
    TopKResampling,
    TopKResamplingConfig,
)


@dataclass
class OrthDiffResamplingConfig(ResamplerConfig):
    gram_schmidt_trail: int = None


class OrthDiffResampling(ResamplingMethod):
    def __init__(self, cfg: OrthDiffResamplingConfig, W_next=None, **kwargs):
        super().__init__(cfg=cfg, W_next=W_next, **kwargs)
        self.cfg = cfg

    def get_directions(self, cache, x, y_pred, y):
        return re_init_neurons_gram_shmidt_precise_iterative_argmax(
            y - y_pred, self.cfg.num_to_resample, self.cfg.gram_schmidt_trail
        )


class OrthRankedTopkResamplingConfig(
    OrthDiffResamplingConfig, TopKResamplingConfig
): ...


class OrthRankedTopkResampling(TopKResampling):
    def get_topk_indices(self, cache, x, y_pred, y):
        directions, indices, x_diff = (
            re_init_neurons_gram_shmidt_precise_iterative_argmax(
                y - y_pred,
                num_to_return=self.cfg.resample_top_k or self.cfg.num_to_resample,
                T=self.cfg.gram_schmidt_trail,
                return_extras=True,
            )
        )
        return indices

    def get_ranking_metric(self, cache, x, y_pred, y):
        assert False, "This method should not be called."
