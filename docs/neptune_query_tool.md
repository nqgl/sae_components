# Neptune Query Tool

A Python API for querying Neptune runs with metric-based filtering and top-k selection.

## Overview

The `NeptuneQuery` class provides an easy-to-use interface for:
- Fetching all runs from a Neptune project
- Filtering runs based on metric constraints
- Finding top-k runs according to a target metric
- Supporting different metric aggregation strategies (last value, mean of last N, min, max, mean all)

## Installation

The tool is located in `src/saeco/analysis/neptune_query.py` and requires the `neptune` package:

```bash
pip install neptune
```

Make sure your Neptune API token is set:
```bash
export NEPTUNE_API_TOKEN="your_api_token_here"
```

## Quick Start

```python
from saeco.analysis.neptune_query import NeptuneQuery

# Initialize query for your project
nq = NeptuneQuery("nqgl/default-project")

# Find top 5 runs with lowest loss
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True
)

# Print results
for run_info in results:
    print(f"Run: {run_info.name}")
    print(f"Loss: {run_info.metrics['train/loss']}")

# Clean up
nq.cleanup()
```

## API Reference

### NeptuneQuery

```python
NeptuneQuery(project: str, api_token: Optional[str] = None)
```

Initialize a Neptune query interface.

**Parameters:**
- `project` (str): Neptune project in format "workspace/project-name"
- `api_token` (Optional[str]): Neptune API token (defaults to NEPTUNE_API_TOKEN env var)

### query_topk

```python
query_topk(
    metric_key: str,
    k: int = 5,
    minimize: bool = True,
    constraints: Optional[dict[str, Callable[[float], bool]]] = None,
    aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
    n_steps: int = 10,
    state: Optional[str] = None
) -> list[RunInfo]
```

Query top-k runs based on a metric, subject to constraints.

**Parameters:**
- `metric_key` (str): Metric to optimize (e.g., "train/loss", "metrics/L0")
- `k` (int): Number of top runs to return
- `minimize` (bool): If True, return runs with lowest metric values; if False, highest
- `constraints` (Optional[dict]): Dict mapping metric keys to constraint functions
  - Example: `{"metrics/L0": lambda x: x < 50, "metrics/L1": lambda x: x > 0.1}`
- `aggregation` (str): How to aggregate metric values:
  - `"last"`: Use the last recorded value
  - `"mean_last_n"`: Average of last N steps
  - `"min"`: Minimum value across all steps
  - `"max"`: Maximum value across all steps
  - `"mean_all"`: Average across all steps
- `n_steps` (int): Number of steps for "mean_last_n" aggregation
- `state` (Optional[str]): Filter by run state ('active', 'inactive', etc.)

**Returns:**
- List of `RunInfo` objects sorted by metric value

### get_run_details

```python
get_run_details(
    run_info: RunInfo,
    metric_keys: Optional[list[str]] = None,
    aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
    n_steps: int = 10
) -> dict[str, Any]
```

Get detailed information about a run.

**Parameters:**
- `run_info` (RunInfo): RunInfo object from query_topk
- `metric_keys` (Optional[list[str]]): Additional metric keys to fetch
- `aggregation` (str): How to aggregate metric values
- `n_steps` (int): Number of steps for "mean_last_n" aggregation

**Returns:**
- Dict with keys: `run_id`, `name`, `config`, `metrics`

### cleanup

```python
cleanup()
```

Close all cached Neptune run connections. Should be called when done querying.

## RunInfo Object

Each run is represented by a `RunInfo` object with the following fields:

- `run_id` (str): Neptune run ID
- `name` (str): Run name
- `config` (dict): Run configuration/parameters
- `metrics` (dict): Metric values extracted during query
- `raw_run`: Neptune Run object (for advanced usage)

## Usage Examples

### Example 1: Simple Top-K Query

Find the 5 runs with the lowest training loss:

```python
from saeco.analysis.neptune_query import NeptuneQuery

nq = NeptuneQuery("nqgl/default-project")

results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True
)

for i, run_info in enumerate(results, 1):
    print(f"{i}. {run_info.name}: loss={run_info.metrics['train/loss']:.4f}")

nq.cleanup()
```

### Example 2: Constrained Query

Find runs with lowest loss where L0 < 50:

```python
nq = NeptuneQuery("nqgl/default-project")

results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    constraints={
        "metrics/L0": lambda x: x < 50
    }
)

for run_info in results:
    print(f"Run: {run_info.name}")
    print(f"  Loss: {run_info.metrics['train/loss']:.4f}")
    print(f"  L0: {run_info.metrics['metrics/L0']:.2f}")

nq.cleanup()
```

### Example 3: Multiple Constraints

Find runs with lowest loss where L0 < 50 AND L1 > 0.1:

```python
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    constraints={
        "metrics/L0": lambda x: x < 50,
        "metrics/L1": lambda x: x > 0.1
    }
)
```

### Example 4: Using Different Aggregations

Use average of last 10 steps instead of just the last value:

```python
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    aggregation="mean_last_n",
    n_steps=10
)
```

Find runs with the best minimum loss ever achieved:

```python
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    aggregation="min"
)
```

### Example 5: Finding Best Performance

Find runs with highest validation accuracy:

```python
results = nq.query_topk(
    metric_key="eval/accuracy",
    k=10,
    minimize=False,  # We want to MAXIMIZE accuracy
    aggregation="max"  # Best accuracy achieved
)
```

### Example 6: Getting Additional Metrics

Get detailed info with additional metrics:

```python
results = nq.query_topk(
    metric_key="train/loss",
    k=3,
    minimize=True
)

# Get more details about the best run
best_run = results[0]
details = nq.get_run_details(
    best_run,
    metric_keys=["eval/accuracy", "metrics/L0", "metrics/L1", "train/loss"],
    aggregation="last"
)

print(f"Best Run: {details['name']}")
print(f"Config: {details['config']}")
print(f"All Metrics: {details['metrics']}")
```

### Example 7: Complex Constraint Logic

Use custom constraint functions with complex logic:

```python
def l0_constraint(x):
    return 30 < x < 60  # L0 must be between 30 and 60

def loss_threshold(x):
    return x < 0.5  # Loss must be below 0.5

results = nq.query_topk(
    metric_key="eval/accuracy",
    k=5,
    minimize=False,  # Maximize accuracy
    constraints={
        "metrics/L0": l0_constraint,
        "train/loss": loss_threshold
    },
    aggregation="mean_last_n",
    n_steps=20
)
```

## Integration with Existing Code

You can use this tool alongside your existing Neptune logging:

```python
from saeco.mlog.mlog_ import mlog
from saeco.analysis.neptune_query import NeptuneQuery

# Your training code with Neptune logging
mlog.init(project="nqgl/default-project", run_name="my_experiment")
mlog.log({"train/loss": 0.5}, step=100)
# ... more training ...
mlog.finish()

# Later, query the best runs
nq = NeptuneQuery("nqgl/default-project")
best_runs = nq.query_topk("train/loss", k=5, minimize=True)

# Extract configs from best runs for further experiments
best_configs = [run.config for run in best_runs]
```

## Notes

- The tool caches fetched runs to avoid repeated API calls
- Use `force_refresh=True` in `fetch_runs()` to bypass cache
- Always call `cleanup()` when done to close Neptune connections
- Metric keys should match the paths used when logging (e.g., "train/loss", "metrics/L0")
- The tool is read-only and won't modify any runs

## Troubleshooting

**Issue:** `None` values in metrics
- **Solution:** Check that the metric key matches exactly what was logged. Neptune paths are case-sensitive.

**Issue:** No runs returned despite valid constraints
- **Solution:** Some runs may not have all metrics logged. The tool filters out runs missing required metrics.

**Issue:** Slow performance
- **Solution:** Fetching many runs with many metrics can be slow. Consider filtering by run state or using cached results.

**Issue:** API token errors
- **Solution:** Make sure `NEPTUNE_API_TOKEN` environment variable is set or pass `api_token` parameter.
