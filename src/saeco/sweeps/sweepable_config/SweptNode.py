from saeco.sweeps.sweepable_config.SweepExpression import SweepExpression
from saeco.sweeps.sweepable_config.Swept import Swept
from saeco.sweeps.sweepable_config.sweep_expressions import SweepVar
from saeco.sweeps.sweepable_config.shared_fns import has_sweep


from pydantic import BaseModel, Field


import random

Location = list[str]


class SweptNode(BaseModel):
    location: Location = Field(default_factory=list)
    children: dict[str, "SweptNode"] = Field(default_factory=dict)
    swept_fields: dict[str, Swept] = Field(default_factory=dict)
    expressions: dict[str, SweepExpression] = Field(default_factory=dict)
    # sweepvars: set[SweepVar] = Field(default_factory=set)

    @classmethod
    def from_sweepable(
        cls,
        target: BaseModel | dict,
        location: Location = [],
    ) -> "SweptNode":
        inst = cls(location=location)
        if isinstance(target, BaseModel):
            items = [(k, getattr(target, k)) for (k, v) in target.model_fields.items()]
        else:
            assert isinstance(target, dict)
            items = target.items()
        for name, attr in items:
            if isinstance(attr, SweepExpression):
                inst.expressions[name] = attr
                # inst.sweepvars |= attr.get_sweepvars()
            elif isinstance(attr, SweepVar):
                raise NotImplementedError("sweepvars should be inside an expression")
                # inst.sweepvars.add(attr)
            elif isinstance(attr, Swept):
                inst.swept_fields[name] = attr
            elif isinstance(attr, BaseModel | dict) and has_sweep(attr):
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

    def swept_combinations_count(self):
        """excluding sweepvar options"""
        n = 1
        for k, v in self.swept_fields.items():
            n *= len(v.values)
        for k, v in self.children.items():
            n *= v.swept_combinations_count()
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

    def get_paths_to_sweep_expressions(self) -> list[Location]:
        paths = []
        for k, v in self.children.items():
            paths.extend([[k] + p for p in v.get_paths_to_sweep_expressions()])
        paths.extend([[k] for k in self.expressions.keys()])
        return paths
