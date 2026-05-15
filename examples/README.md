# Examples

Curated demos of the public `saeco` API. Each file is standalone and
explains what it demonstrates in its top-of-file docstring.

| File | Demonstrates | Runs without GPU? |
|------|--------------|-------------------|
| [`define_architecture.py`](define_architecture.py) | Defining a custom `Architecture` subclass: config, model, losses, setup wrappers/mixins | Yes |
| [`sweep_dsl.py`](sweep_dsl.py) | The `SweepableConfig` DSL: `Swept`, `SweepVar`, `SweepExpression`, `Val`, combination counting | Yes |
| [`train_vanilla_sae.py`](train_vanilla_sae.py) | Wiring a full `RunConfig` and running a training job (single config or full sweep) | No (needs CUDA) |

## Running

After installing the package (`pip install -e .` from the repo root):

```bash
python examples/define_architecture.py
python examples/sweep_dsl.py
python examples/train_vanilla_sae.py     # GPU + downloaded GPT-2
```

## See also

- [`research/experiments/`](../research/experiments/) — larger, in-flight
  training scripts. Less curated; some are broken; useful as a working
  reference for "how do I actually do X?"
- [`research/scripts/`](../research/scripts/) — exploration and analysis
  scripts.
