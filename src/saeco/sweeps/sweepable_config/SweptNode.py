from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.sweeps.sweepable_config.has_sweep import CouldHaveSweep, has_sweep, to_items
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field


import random

Location = list[str]

T = TypeVar("T")


class SweptNode(BaseModel):
    location: Location = Field(default_factory=list)
    children: dict[str, "SweptNode"] = Field(default_factory=dict)
    swept_fields: dict[str, Swept] = Field(default_factory=dict)
    expressions: dict[str, SweepExpression] = Field(default_factory=dict)
    # sweepvars: set[SweepVar] = Field(default_factory=set)

    @classmethod
    def from_sweepable(
        cls,
        target: CouldHaveSweep,
        location: Location = [],
    ) -> "SweptNode":
        inst = cls(location=location)
        for name, attr in to_items(target):
            if isinstance(attr, SweepExpression):
                inst.expressions[name] = attr
                # inst.sweepvars |= attr.get_sweepvars()
            elif isinstance(attr, SweepVar):
                raise NotImplementedError("sweepvars should be inside an expression")
                # inst.sweepvars.add(attr)
            elif isinstance(attr, Swept):
                inst.swept_fields[name] = attr
            elif isinstance(attr, CouldHaveSweep) and has_sweep(attr):
                inst.children[name] = cls.from_sweepable(attr, location + [name])
            else:
                continue
        return inst

    def get_sweepvars(self) -> set[SweepVar]:
        s = set()
        for k, v in self.expressions.items():
            s |= v.get_sweepvars()
        for k, v in self.children.items():
            s |= v.get_sweepvars()
        return s

    def get_vars_dict(self) -> dict[str, SweepVar]:
        sweepvars = self.get_sweepvars()
        d = {}
        for var in sweepvars:
            if var.name in d and d[var.name] != var:
                raise ValueError(f"SweepVar {var.name} is not unique/consistent")
            d[var.name] = var
        return d

    def _swept_combinations_count_vars_only(self):
        sweepvars = self.get_vars_dict()
        n = 1
        for var in sweepvars.values():
            n *= len(var.values)
        return n

    def swept_combinations_count_including_vars(self):
        v = (
            self._swept_combinations_count_vars_only()
            * self._swept_combinations_count_excluding_vars()
        )
        assert 0 < v and isinstance(
            v, int
        ), "swept_combinations_count should be positive"
        return v

    def _swept_combinations_count_children_count(self):
        n = 1
        for k, v in self.children.items():
            n *= v._swept_combinations_count_excluding_vars()
        return n

    def _swept_combinations_count_directly_contained_swept_fields_count(self):
        n = 1
        for k, v in self.swept_fields.items():
            n *= len(v.values)
        return n

    def _swept_combinations_count_excluding_vars(self):
        """excluding sweepvar options
        this is the number of directly contained
        """
        n = 1
        for k, v in self.swept_fields.items():
            n *= len(v.values)
        for k, v in self.children.items():
            n *= v._swept_combinations_count_excluding_vars()
        return n

    def swept_options_sum(self):
        n = 0
        for k, v in self.swept_fields.items():
            n += len(v.values)
        for k, v in self.children.items():
            n += v.swept_options_sum()
        return n

    def to_wandb(self):
        sweepvars = self.get_sweepvars()
        return {
            "parameters": {
                "sweep_vars": {
                    "parameters": {k.name: k.sweep_dump() for k in sweepvars}
                },
                **{
                    k: v._to_wandb_parameters_only()
                    for k, v in self.children.items()
                    if v.swept_options_sum() > 0
                },
                **{k: v.model_dump() for k, v in self.swept_fields.items()},
            },
            "method": "grid",
        }

    def _to_wandb_parameters_only(self):
        return {
            "parameters": {
                **{k: v._to_wandb_parameters_only() for k, v in self.children.items()},
                **{k: v.model_dump() for k, v in self.swept_fields.items()},
            }
        }

    def random_selection(self, sweep_vars=None):
        if sweep_vars is None:
            vars = self.get_sweepvars()
            var_values = {var.name: random.choice(var.values) for var in vars}
            return {
                **self.random_selection(var_values),
                "sweep_vars": var_values,
            }

        return {
            **{k: v.random_selection(sweep_vars) for k, v in self.children.items()},
            **{k: random.choice(v.values) for k, v in self.swept_fields.items()},
        }

    @classmethod
    def consistent_sort(cls, d: T) -> T:
        if isinstance(d, dict):
            return cls.alphabetize_dict(d)
        elif isinstance(d, list):
            return sorted([cls.consistent_sort(i) for i in d])
        else:
            return d

    @classmethod
    def alphabetize_dict(cls, d: dict[str, T]) -> dict[str, T]:
        return {k: v for k, v in sorted(d.items(), key=lambda x: x[0])}

    def select_instance_by_index(self, i, sweep_vars=None) -> dict[str, Any]:
        if not 0 <= i < self.swept_combinations_count_including_vars():
            raise IndexError("i should be less than the number of combinations")
        if sweep_vars is None:
            i_vars = i // self._swept_combinations_count_excluding_vars()
            i_children = i % self._swept_combinations_count_excluding_vars()
            var_values = {}
            for k, v in self.consistent_sort(self.get_vars_dict()).items():
                var_values[v.name] = v.values[i_vars % len(v.values)]
                i_vars //= len(v.values)
            assert i_vars == 0, "i_vars should be 0"
            return {
                **self.select_instance_by_index(i_children, sweep_vars=var_values),
                "sweep_vars": var_values,
            }
        d = {
            k: v.evaluate(sweep_vars)
            for k, v in self.consistent_sort(self.expressions).items()
        }
        for k, v in self.consistent_sort(self.swept_fields).items():
            assert k not in d
            d[k] = v.values[i % len(v.values)]
            i //= len(v.values)
        assert (
            i < self._swept_combinations_count_children_count()
        ), "i should be less than the number of children combinations"
        for k, v in self.consistent_sort(self.children).items():
            assert k not in d
            d[k] = v.select_instance_by_index(
                i % v._swept_combinations_count_excluding_vars(),
                sweep_vars=sweep_vars,
            )
            i //= v._swept_combinations_count_excluding_vars()
        assert i == 0, "i should be 0"
        return d

    def get_paths_to_sweep_expressions(self) -> list[Location]:
        paths = []
        for k, v in self.children.items():
            paths.extend([[k] + p for p in v.get_paths_to_sweep_expressions()])
        paths.extend([[k] for k in self.expressions.keys()])
        return paths

    def _to_optuna_grid_search_space_params_only(self, values_only=True):
        d = {
            f"{k}/{child_k}": child_v
            for k, v in self.children.items()
            for child_k, child_v in v._to_optuna_grid_search_space_params_only(
                values_only=values_only
            ).items()
        }
        d.update(
            {k: v.values if values_only else v for k, v in self.swept_fields.items()}
        )
        return d

    def to_optuna_grid_search_space(self, values_only=True):
        sweepvars = self.get_sweepvars()
        d = self._to_optuna_grid_search_space_params_only()
        sweepvars_d = {
            f"sweep_vars/{sv.name}": sv.values if values_only else sv
            for sv in sweepvars
        }
        return {**d, **sweepvars_d}


def main():
    from saeco.sweeps.sweepable_config import (
        SweepableConfig,
        Swept,
        SweepExpression,
        SweepVar,
    )
    from saeco.architectures.vanilla import VanillaConfig, VanillaSAE

    class Ex(SweepableConfig):
        a: int = 1
        b: int = 2
        c: int = 3

    var = SweepVar(name="var", values=[1, 2, 3])

    cfg = Ex(a=var * 5, b=2, c=Swept(10, 100, 1_000, 10_000))
    check_config_combinations(cfg)


if TYPE_CHECKING:
    from saeco.sweeps.sweepable_config import SweepableConfig


def check_config_combinations(cfg: "SweepableConfig"):
    nodes = cfg.to_swept_nodes()
    l = []
    for i in range(nodes.swept_combinations_count_including_vars()):
        l.append(nodes.select_instance_by_index(i))
    for i in range(len(l)):
        for j in range(len(l)):
            if i == j:
                continue
            assert l[i] != l[j]
            assert l[i] == cfg.to_swept_nodes().select_instance_by_index(i)
    print("combination count:", nodes.swept_combinations_count_including_vars())
    print("vars:", nodes.get_sweepvars())

    try:
        nodes.select_instance_by_index(len(l))
        raise ValueError("should raise error")
    except IndexError as e:
        print("error correctly raised:", e)


if __name__ == "__main__":
    main()
