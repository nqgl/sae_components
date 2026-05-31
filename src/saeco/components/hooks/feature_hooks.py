import torch

from saeco.components.features.features import HasFeatures
from saeco.components.features.features_param import FeatureParamType
from saeco.components.type_acc_methods import post_backward_hook, post_step_hook
from saeco.components.wrap import WrapsModule


@torch.no_grad()
def _normalize_features(
    module: HasFeatures,
    *,
    which: FeatureParamType,
    norm_ord: float,
    max_only: bool,
) -> None:
    fp = module.features.get(which)
    norm = torch.linalg.vector_norm(fp.features, dim=-1, keepdim=True, ord=norm_ord)
    if (norm == 0).any():
        print("Norm is zero, not normalizing.")
        return
    if not max_only:
        fp.features[:] = fp.features / norm
    else:
        fp.features[:] = torch.where(norm > 1, fp.features / norm, fp.features)


@torch.no_grad()
def _orthogonalize_feature_grads(
    module: HasFeatures, *, which: FeatureParamType
) -> None:
    fp = module.features.get(which)
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

    Wraps any :class:`HasFeatures` module. ``which`` selects which
    :class:`FeatureParamType` of ``wrapped.features`` to normalize (default:
    the decoder param). ``norm_ord`` chooses the vector norm; ``max_only=True``
    clips to unit norm only those features whose norm exceeds 1.
    """

    def __init__(
        self,
        wrapped: HasFeatures,
        which: FeatureParamType = FeatureParamType.dec,
        norm_ord: float = 2,
        max_only: bool = False,
    ):
        super().__init__(wrapped)
        self._self_which = FeatureParamType(which)
        self._self_ord = norm_ord
        self._self_max_only = max_only

    @post_step_hook
    def normalize_features(self) -> None:
        _normalize_features(
            self.__wrapped__,
            which=self._self_which,
            norm_ord=self._self_ord,
            max_only=self._self_max_only,
        )


class OrthogonalizeFeatureGrads(WrapsModule):
    """Project decoder feature gradients onto the unit-sphere tangent space.

    Used together with :class:`NormFeatures`: keeps the gradient step in the
    surface where feature norms remain constant, instead of fighting the
    unit-norm constraint. ``which`` selects which :class:`FeatureParamType` to
    orthogonalize (default: the decoder param).
    """

    def __init__(
        self, wrapped: HasFeatures, which: FeatureParamType = FeatureParamType.dec
    ):
        super().__init__(wrapped)
        self._self_which = FeatureParamType(which)

    @post_backward_hook
    def orthogonalize_feature_grads(self) -> None:
        _orthogonalize_feature_grads(self.__wrapped__, which=self._self_which)
