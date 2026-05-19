# sweepable

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

Pydantic configs that double as hyperparameter sweep specifications. Each
field is implicitly `T | Swept[T] | SweepExpression`; a runner can
enumerate the resulting concrete configurations.

## Installation

```bash
pip install sweepable
```

(Or `pip install -e .` from this directory for development.)

## Quickstart

```python
from sweepable import SweepableConfig, SweepVar, Swept, Val


class TrainCfg(SweepableConfig):
    lr: float = 1e-3
    batch_size: int = 4096
    num_steps: int = 50_000
    pre_bias: bool = False


# A concrete config — represents a single run.
single = TrainCfg()
print(single.is_concrete())  # True

# A grid: 3 lr values × 2 pre_bias values = 6 runs.
grid = TrainCfg(
    lr=Swept(1e-3, 3e-4, 1e-4),
    pre_bias=Swept(True, False),
)
print(grid.sweep_info_tree.swept_combinations_count_including_vars())  # 6
print(grid.random_sweep_configuration())  # one randomly-selected concrete config

# A SweepVar is a *named* sweep axis that can be referenced from
# multiple fields. Each value produces one run regardless of how many
# fields it feeds into, so coupled fields move together on one axis.
# Here: hold the token budget (512 * 2**16) fixed while varying batch
# size — a compute-allocation equivalence sweep.
w = SweepVar(1, 2, 4, 8, 16, name="batch_size_weight")
budget = TrainCfg(
    batch_size=Val(value=512) * w,
    num_steps=Val(value=2**16) // w,
)
print(budget.sweep_info_tree.swept_combinations_count_including_vars())  # 5
```

## Public API

```python
from sweepable import (
    SweepableConfig,   # Pydantic BaseModel subclass with sweep semantics
    Swept,             # one-axis sweep over a list of values
    SweepVar,          # named sweep axis, shareable across fields
    SweepExpression,   # compound expression (Op tree)
    Val,               # constant wrapped for use in expressions
)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Provenance

This package was extracted from the [`saeco`](https://github.com/nqgl/sae_components)
codebase, where it serves as the configuration backbone for sparse
autoencoder training sweeps. It has no saeco-specific dependencies and
stands alone.
