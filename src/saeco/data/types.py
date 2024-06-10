from typing import List
from torch import Tensor
from jaxtyping import Int, Float

TextData = List[str] | str
TokensData = List[List[int]] | List[int] | Int[Tensor, "batch seq_len"]
ActsDataSeq = Float[Tensor, "batch seq d_data"]
ActsData = ActsDataSeq | Float[Tensor, "batch d_data"]
