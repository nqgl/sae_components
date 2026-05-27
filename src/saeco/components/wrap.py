from typing import cast

import torch.nn as nn
import wrapt


# T should be nn.Module bound but bc no intersection type may have to do no bound sadly
class WrapsModule[T: nn.Module = nn.Module](wrapt.CallableObjectProxy[T]):
    """Transparent proxy around an :class:`nn.Module`.

    The only intentionally non-transparent method is ``apply``: PyTorch's
    ``Module.apply`` would otherwise visit only the wrapped module tree, so
    wrapper-defined training hooks would be invisible to the trainer.
    """

    def __init__(self, wrapped: T):
        super().__init__(wrapped)

    def apply(self, fn):
        """Visit the wrapper itself in addition to the wrapped module.

        ``nn.Module.apply`` recurses through ``_modules`` and ends with
        ``fn(self)``. A wrapt proxy forwards ``_modules`` and ``apply`` to
        the wrapped module, so the wrapper would otherwise be invisible to
        ``apply()`` — and the training loop discovers ``post_step_hook`` /
        ``post_backward_hook`` methods via ``apply()``. We traverse the
        wrapped's children first (matching the order of ``nn.Module.apply``)
        and then call ``fn(self)`` so any wrapper-defined hooks are seen.
        """
        self.__wrapped__.apply(fn)
        fn(self)
        return self

    def _get_name(self) -> str:
        # for nn.Module __repr__
        return f"{type(self).__name__}[{self.__wrapped__._get_name()}]"

    def __repr__(self) -> str:
        # Defer to nn.Module's repr (which uses _get_name + the wrapped's
        # children), so stacked wrappers print nicely.
        return nn.Module.__repr__(cast(nn.Module, self))
