# %%
from typing_extensions import Unpack
from pydantic._internal._generics import PydanticGenericMetadata
import wandb
from typing import TypeVar, Generic, Type, List, Annotated, Union
from pydantic import BaseModel, create_model, dataclasses

PROJECT = "sweep_test"


def initialize_sweep(d):
    sweep_id = wandb.sweep(
        sweep=d,
        project=PROJECT,
    )
    f = open("sweep/sweep_id.txt", "w")
    f.write(sweep_id)
    f.close()


T = TypeVar("T")


class Swept(BaseModel, Generic[T]):
    values: list[T]

    def __init__(self, *values, **kwargs):
        if values:
            if "values" in kwargs:
                raise Exception("two sources of values in Swept initialization!")
            kwargs["values"] = values
        return super().__init__(**kwargs)


class Category(BaseModel):
    name: str
    extra_values: 


class CategSweep(BaseModel):
    category: Swept[Category]


cs = CategSweep(
    category=Swept[Category](
        values=[Category(name="VanillaSAE"), Category(name="HierarchicalSAE")]
    )
)
d = cs.model_dump()
dj = cs.model_dump_json()
cs2 = CategSweep.model_validate(d)
cs2.category.values[0]

T_CFG = TypeVar("T_CFG")


class SweptConfig(BaseModel, Generic[T_CFG]):
    @classmethod
    def __class_getitem__(cls, t_cfg: T) -> T_CFG:
        fields = {
            name: (Union[Swept[field.annotation], field.annotation], ...)
            for name, field in t_cfg.model_fields.items()
        }

        return create_model(t_cfg.__name__, __base__=t_cfg, **fields)


ExampleBaseConfig: SweptConfig[ExampleBaseConfig]

# %%
import pydantic._internal._model_construction as mc
from typing import Any


def Sweepable(type):
    assert not isinstance(
        type, Swept
    ), "Swept type should not be wrapped in Sweepable or passed to SweepableConfig"
    return Union[Swept[type], type]


class SweptMeta(mc.ModelMetaclass):
    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        __pydantic_generic_metadata__=None,
        __pydantic_reset_parent_namespace__: bool = True,
        _create_model_module: str | None = None,
        **kwargs: Any,
    ) -> type:
        namespace["__annotations__"] = {
            name: Sweepable(annotation)
            for name, annotation in namespace["__annotations__"].items()
        }
        return super().__new__(
            mcs,
            cls_name,
            bases,
            namespace,
            __pydantic_generic_metadata__,
            __pydantic_reset_parent_namespace__,
            _create_model_module,
            **kwargs,
        )
class SweepableConfig(BaseModel, metaclass=SweptMeta):
    __ignore_this: int = 0

    def is_concrete(self, search_target:BaseModel=None):
        search_target = search_target or self
        for name, field in search_target.model_fields.items():
            attr = getattr(search_target, name)
            if isinstance(attr, Swept):
                return False
            elif isinstance(attr, BaseModel):
                if not self.is_concrete(attr):
                    return False
        return True


class Cfg(SweepableConfig):
    an_int: int
    a_string: str = Swept(["a", "b"])
    a_float: float = 0.1

class Cfg2(Cfg):
    b: int

c21 = Cfg2(an_int=1, a_string="b", b=2)
c2 = Cfg2(an_int=Swept[int](3,4), a_string="b", b=Swept[int](values=[1, 2, 3]))

# %%
cfg=B(c=Cfg(an_int=1, a_string="b"))
# cfg.is_concrete()

cfg2 = Cfg2(cfg=cfg, an_int=2)
cfg2.is_concrete()
# ff
# %%
def run():
    wandb.init()
    scfg = ConfigFromSweep(**wandb.config)
    cfg, lgcfg = get_configs_from_sweep(scfg=scfg)
    # if cfg.sae_cfg.sae_type != "VanillaSAE":
    #     cfg.neuron_dead_threshold = -1
    cfg.l1_coeff = cfg.l1_coeff * sparsity_coeff_adjustment(scfg)
    wandb.config.update({"adjusted_l1_coeff": cfg.l1_coeff})

    nice_name = wandb.config["sae_type"]
    wandb.finish()

    l0_target = 45
    cfg = TrainConfig(
        l0_target=l0_target,
        coeffs={
            "sparsity_loss": 2e-3 if l0_target is None else 3e-4,
            "L2_loss": 10,
        },
        lr=1e-3,
        use_autocast=True,
        wandb_cfg=dict(project=PROJECT),
        l0_target_adjustment_size=0.001,
        batch_size=2048,
        use_lars=True,
        betas=(0.9, 0.99),
    )


# from torch.utils.viz._cycles import warn_tensor_cycles
#
# warn_tensor_cycles()

from dataclasses import dataclass, field

sweep_id = open("sweeps/sweep_id.txt").read().strip()

SWEEP_NAME = "sweep_test"


def main():
    wandb.agent(
        sweep_id,
        function=run,
        project=PROJECT,
    )


if __name__ == "__main__":
    main()

# %%
