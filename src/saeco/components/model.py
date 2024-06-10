import saeco.core.module as cl
from abc import abstractmethod
from typing import List
from saeco.components import Loss


class Model(cl.Module):
    def __init__(self, loss_cls_list: List[Loss]):
        super().__init__()
        self.losses = ...
