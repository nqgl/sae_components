# %%
import wandb
from typing import TypeVar, Generic, Type
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


class ExampleBaseConfig(BaseModel):
    an_int: int
    a_float: float
    a_string: str


# %%
T_CFG = TypeVar("T_CFG")


class SweptConfig(BaseModel, Generic[T_CFG]):
    @classmethod
    def __class_getitem__(cls, t_cfg: Type[T_CFG]) -> Type[T_CFG]:
        class SweptConfigImpl(BaseModel):
            def __new__(cls2, **kwargs):
                c = create_model(t_cfg.__name__, __base__=t_cfg, **fields)
                return c

        annotations = t_cfg.__annotations__
        fields = {
            name: (Swept[field.annotation], ...)
            for name, field in t_cfg.model_fields.items()
        }

        return create_model(t_cfg.__name__, __base__=t_cfg, **fields)


T = TypeVar("T")


class ListedFields(Generic[T_CFG], BaseModel):
    @classmethod
    def __class_getitem__(cls, t_cfg: Type[T]) -> Type[T]:
        t = super().__class_getitem__(t_cfg)

        class ListedFieldsImpl(t_cfg):
            def __new__(cls2, **kwargs) -> T_CFG:
                fields = {
                    name: (Swept[field.annotation], ...)
                    for name, field in t_cfg.model_fields.items()
                }
                c = create_model(t_cfg.__name__, __base__=t_cfg, **fields)
                return c(**kwargs)

        # for name, field in t.model_fields.items():
        #     setattr(ListedFieldsImpl, name, list[field.annotation])
        ListedFieldsImpl.__name__ = t_cfg.__name__ + "ListedFields"
        return ListedFieldsImpl


c: ListedFields[ExampleBaseConfig] = ListedFields[ExampleBaseConfig](
    an_int=Swept[int](values=[1, 2, 3]),
    a_float=Swept[float](values=[]),
    a_string=Swept[str](values=[]),
)
c.an_int


# %%


class BaseSweptConfig(BaseModel, Generic[T_CFG]):
    pass


class SweptConfigMeta(type(BaseSweptConfig)):
    def __class_getitem__(cls, t_cfg: Type[BaseModel]) -> BaseSweptConfig[T_CFG]:
        class SweptConfigImpl(BaseModel):
            pass

        super

        for name, field in t_cfg.__fields__.items():
            setattr(SweptConfigImpl, name, Swept[field.annotation])
            print(name, field.annotation)

        SweptConfigImpl.__name__ = t_cfg.__name__ + "SweptConfig"
        return SweptConfigImpl


class SweptConfig2(BaseModel, Generic[T_CFG], metaclass=SweptConfigMeta):
    pass


b: SweptConfig2[ExampleBaseConfig] = SweptConfig2[ExampleBaseConfig](
    an_int=Swept[int](values=[1, 2, 3]),
    a_float=Swept[float](values=[]),
    a_string=Swept[str](values=[]),
)

b.an_int

# %%


from typing import Type, TypeVar, get_type_hints, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def transform_class(cls: Type[T], transform_func: callable) -> Type[T]:
    type_hints = get_type_hints(cls)
    transformed_fields = {
        field_name: transform_func(field_type)
        for field_name, field_type in type_hints.items()
    }
    print(type_hints.items())
    return type(cls.__name__, (cls,), transformed_fields)


class User(BaseModel):
    name: str
    age: int


def transform_to_optional(field_type):
    return Optional[field_type]


OptionalUser = transform_class(User, transform_to_optional)
# %%
from typing import Type, TypeVar, get_type_hints, Optional, cast
from typing_extensions import TypedDict
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# def transform_class(cls: Type[T], transform_func: callable) -> Type[T]:
#     type_hints = get_type_hints(cls)
#     transformed_fields = {
#         field_name: transform_func(field_type)
#         for field_name, field_type in type_hints.items()
#     }

#     # Create a TypedDict for the transformed class
#     TransformedClass = TypedDict('TransformedClass', transformed_fields)

#     # Create the transformed class dynamically
#     transformed_class = type(cls.__name__, (cls,), transformed_fields)

#     # Use cast() to provide a type hint for the transformed class
#     return cast(Type[T], transformed_class)


def transform_class(cls: Type[T], transform_func: callable) -> Type[T]:
    type_hints = get_type_hints(cls)
    transformed_fields = {
        field_name: (transform_func(field_type), field_type)
        for field_name, field_type in type_hints.items()
    }

    # Create the transformed class using create_model
    transformed_class = create_model(cls.__name__, __base__=cls, **transformed_fields)

    # Use cast() to provide a type hint for the transformed class
    return cast(Type[T], transformed_class)


class User(BaseModel):
    name: str
    age: int


def transform_to_optional(field_type):
    return Optional[field_type]


# %%

T = TypeVar("T", bound=BaseModel)


@dataclasses.dataclass_transform
def transform_class(cls: Type[T], transform_func: callable) -> Type[T]:
    type_hints = get_type_hints(cls)
    transformed_fields: Dict[str, Annotated[Any, Any]] = {
        field_name: Annotated[transform_func(field_type), field_type]
        for field_name, field_type in type_hints.items()
    }

    # Create a new type with the transformed fields
    class TransformedModel(BaseModel):
        __annotations__ = transformed_fields

    # Create the transformed class using create_model
    transformed_class = create_model(
        cls.__name__, __base__=TransformedModel, **transformed_fields
    )

    return transformed_class


def transform_to_list(field_type):
    return list[field_type]


OptionalUser = transform_class(User, transform_to_list)

# Example usage
user = OptionalUser(name=["John"], age=23)
user.age


# %%
from typing import Type, TypeVar, get_type_hints, Annotated, List, cast, Dict, Any
from pydantic import BaseModel, create_model

T = TypeVar("T", bound=BaseModel)


def transform_class(cls: Type[T], transform_func: callable) -> Type[BaseModel]:
    type_hints = get_type_hints(cls)
    transformed_fields: Dict[str, Annotated[Any, Any]] = {
        field_name: Annotated[transform_func(field_type), field_type]
        for field_name, field_type in type_hints.items()
    }

    # Create the transformed class using create_model
    transformed_class = create_model(cls.__name__, __base__=cls, **transformed_fields)

    return transformed_class


class User(BaseModel):
    name: str
    age: int


def transform_to_list(field_type):
    return List[field_type]


ListUser = transform_class(User, transform_to_list)

# Example usage
user = ListUser(name=["John", "Doe"], age=[25, 30])
reveal_type(user.name)  # Revealed type is 'builtins.list[builtins.str]'
reveal_type(user.age)  # Revealed type is 'builtins.list[builtins.int]'
# %%
# It should behave like this


def transform_class(cls: Type[User], transform_func: callable) -> Type[T]:
    type_hints = get_type_hints(cls)
    transformed_fields = {
        field_name: (Annotated[transform_func(field_type), field_type], field_type)
        for field_name, field_type in type_hints.items()
    }

    # Create the transformed class using create_model
    transformed_class = create_model(cls.__name__, __base__=cls, **transformed_fields)
    cl2 = cast(User, transformed_class)
    u = cl2(name=["John", "Doe"], age=[25, 30])

    # Use cast() to provide a type hint for the transformed class


# %%
class TargetTypeBehavior:
    an_int: Swept[int]
    a_float: Swept[float]
    a_string: Swept[str]


# I want to define SweptConfig such that

c = SweptConfig[ExampleBaseConfig]()
c.an_int


class Category(BaseModel):
    name: str


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
