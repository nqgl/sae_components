import sae_components.core as cl
from abc import abstractmethod
from typing import List
from sae_components.components import Loss


class Model(cl.Module):
    def __init__(self, loss_cls_list: List[Loss]):
        super().__init__()
        self.losses = ...
