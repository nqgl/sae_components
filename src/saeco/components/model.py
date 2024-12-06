import saeco.core.module as cl
from typing import List
from saeco.components import Loss


class Model(cl.Module):
    def __init__(self, loss_cls_list: List[Loss]):
        super().__init__()
        self.losses = ...


from attrs import define, field


@define
class Architecture:
    model_gen_fn: callable
