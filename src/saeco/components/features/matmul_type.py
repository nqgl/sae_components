import saeco.core as cl
from saeco.components.features.features import ResampledWeight
from saeco.components.wrap import WrapsModule
from typing import Optional

import torch.nn as nn
from torch import Tensor


from abc import abstractmethod


# class MatMulWeights(WrapsModule):
#     wrapped: cl.ops.MatMul

#     def __init__(self, wrapped: cl.ops.MatMul):
#         if isinstance(wrapped, nn.Parameter):
#             wrapped = cl.ops.MatMul(wrapped)
#         assert isinstance(wrapped, cl.ops.MatMul)
#         super().__init__(wrapped)

#     @abstractmethod
#     def features_transform(self, tensor: Tensor) -> Tensor: ...

#     @property
#     def features(self) -> Tensor:
#         return self.features_transform(self.wrapped.right.data)

#     @property
#     def features_grad(self) -> Optional[Tensor]:
#         grad = self.wrapped.right.grad
#         if grad is None:
#             return None
#         return self.features_transform(grad)

#     def resampled(self):
#         return ResampledWeight(self)


# class DecoderWeights(MatMulWeights):
#     def features_transform(self, tensor: Tensor) -> Tensor:
#         return tensor


# class EncoderWeights(MatMulWeights):
#     wrapped: cl.ops.MatMul

#     def features_transform(self, tensor: Tensor) -> Tensor:
#         return tensor.transpose(-2, -1)
