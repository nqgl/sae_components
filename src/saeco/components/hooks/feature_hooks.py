import torch

from saeco.components.features.features import HasFeatures
from saeco.components.features.features_param import FeaturesParam
from saeco.components.type_acc_methods import post_backward_hook, post_step_hook
from saeco.components.wrap import WrapsModule


def _feature_param(module: HasFeatures, index: str | None) -> FeaturesParam:
    """Pick the ``FeaturesParam`` to operate on.

    With ``index=None`` the wrapped module must expose exactly one
    decoder-type feature parameter (the common single-decoder case); otherwise
    ``index`` keys into ``module.features`` directly.
    """
    if index is None:
        fps = list(module.features.values())
        assert len(fps) == 1, (
            f"Expected exactly one FeaturesParam on {type(module).__name__} "
            f"when no index is given; got {len(fps)}"
        )
        fp = fps[0]
        assert fp.type == "dec", (
            f"Default-index path expects a decoder FeaturesParam; got type={fp.type!r}"
        )
        return fp
    return module.features[index]


@torch.no_grad()
def _normalize_features(
    module: HasFeatures,
    *,
    index: str | None,
    norm_ord: float,
    max_only: bool,
) -> None:
    fp = _feature_param(module, index)
    norm = torch.linalg.vector_norm(fp.features, dim=-1, keepdim=True, ord=norm_ord)
    if (norm == 0).any():
        print("Norm is zero, not normalizing.")
        return
    if not max_only:
        fp.features[:] = fp.features / norm
    else:
        fp.features[:] = torch.where(norm > 1, fp.features / norm, fp.features)


@torch.no_grad()
def _orthogonalize_feature_grads(module: HasFeatures, *, index: str | None) -> None:
    fp = _feature_param(module, index)
    if fp.grad is None:
        return

    dec_normed = fp.features / fp.features.norm(dim=-1, keepdim=True)
    grad_orth = fp.grad - (dec_normed * fp.grad).sum(-1, keepdim=True) * dec_normed
    test = grad_orth * dec_normed + fp.grad
    if fp.grad.isinf().any():
        print("Infs in grads! ignoring.")
    if fp.grad.isnan().any():
        print("NaNs in grads! returning")
        return
    if test.isinf().any():
        print("Infs in test! ignoring.")
    if test.isnan().any():
        print("NaNs in test! returning")
        return
    assert (grad_orth / (grad_orth.norm(dim=-1, keepdim=True) + 1e-6) * dec_normed).sum(
        -1
    ).abs().mean() < 1e-4, (
        "Not orthogonal, oops. How not orthogonal? This much (max): "
        f"{(fp.grad * fp.features).sum(-1).abs().max()}"
    )
    fp.grad[:] = grad_orth


class NormFeatures(WrapsModule):
    """Renormalize a layer's features to unit norm after each optimizer step.

    Wraps any :class:`HasFeatures` module. ``index`` selects which entry of
    ``wrapped.features`` to normalize (default: the sole decoder-type
    parameter). ``norm_ord`` chooses the vector norm; ``max_only=True``
    clips to unit norm only those features whose norm exceeds 1.
    """

    def __init__(
        self,
        wrapped: HasFeatures,
        index: str | None = None,
        norm_ord: float = 2,
        max_only: bool = False,
    ):
        super().__init__(wrapped)
        self._self_index = index
        self._self_ord = norm_ord
        self._self_max_only = max_only

    @post_step_hook
    def normalize_features(self) -> None:
        _normalize_features(
            self.__wrapped__,
            index=self._self_index,
            norm_ord=self._self_ord,
            max_only=self._self_max_only,
        )


class OrthogonalizeFeatureGrads(WrapsModule):
    """Project decoder feature gradients onto the unit-sphere tangent space.

    Used together with :class:`NormFeatures`: keeps the gradient step in the
    surface where feature norms remain constant, instead of fighting the
    unit-norm constraint. ``index`` selects which ``FeaturesParam`` to
    orthogonalize (default: the sole decoder-type parameter).
    """

    def __init__(self, wrapped: HasFeatures, index: str | None = None):
        super().__init__(wrapped)
        self._self_index = index

    @post_backward_hook
    def orthogonalize_feature_grads(self) -> None:
        _orthogonalize_feature_grads(self.__wrapped__, index=self._self_index)
