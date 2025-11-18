"""
Example usage of the Neptune query tool.

This script demonstrates how to use the NeptuneQuery class to:
1. Fetch runs from a Neptune project
2. Filter runs based on metric constraints
3. Find top-k runs according to a target metric
"""

from saeco.analysis.neptune_query import NeptuneQuery

# Initialize the query tool for your Neptune project
nq = NeptuneQuery("nqgl/default-project")

# Example 1: Find top 5 runs with lowest loss
print("=" * 80)
print("Example 1: Top 5 runs with lowest loss")
print("=" * 80)

results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    aggregation="last"  # Use the last recorded value
)

for i, run_info in enumerate(results, 1):
    print(f"\n{i}. Run: {run_info.name}")
    print(f"   ID: {run_info.run_id}")
    print(f"   Loss: {run_info.metrics.get('train/loss', 'N/A')}")

# Example 2: Find top 5 runs with lowest loss, subject to L0 < 50
print("\n" + "=" * 80)
print("Example 2: Top 5 runs with lowest loss where L0 < 50")
print("=" * 80)

results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    constraints={
        "metrics/L0": lambda x: x < 50  # L0 must be less than 50
    },
    aggregation="last"
)

for i, run_info in enumerate(results, 1):
    print(f"\n{i}. Run: {run_info.name}")
    print(f"   ID: {run_info.run_id}")
    print(f"   Loss: {run_info.metrics.get('train/loss', 'N/A')}")
    print(f"   L0: {run_info.metrics.get('metrics/L0', 'N/A')}")

# Example 3: Find runs with multiple constraints
print("\n" + "=" * 80)
print("Example 3: Top 3 runs with lowest loss where L0 < 50 and L1 > 0.1")
print("=" * 80)

results = nq.query_topk(
    metric_key="train/loss",
    k=3,
    minimize=True,
    constraints={
        "metrics/L0": lambda x: x < 50,
        "metrics/L1": lambda x: x > 0.1
    },
    aggregation="last"
)

for i, run_info in enumerate(results, 1):
    print(f"\n{i}. Run: {run_info.name}")
    print(f"   ID: {run_info.run_id}")
    print(f"   Metrics: {run_info.metrics}")
    print(f"   Config: {run_info.config}")

# Example 4: Using average of last N steps instead of just the last value
print("\n" + "=" * 80)
print("Example 4: Top 5 runs by average loss over last 10 steps")
print("=" * 80)

results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    aggregation="mean_last_n",
    n_steps=10
)

for i, run_info in enumerate(results, 1):
    print(f"\n{i}. Run: {run_info.name}")
    print(f"   ID: {run_info.run_id}")
    print(f"   Avg Loss (last 10 steps): {run_info.metrics.get('train/loss', 'N/A')}")

# Example 5: Get detailed information about specific runs
print("\n" + "=" * 80)
print("Example 5: Get detailed information about a run")
print("=" * 80)

if results:
    run_info = results[0]
    details = nq.get_run_details(
        run_info,
        metric_keys=["train/loss", "metrics/L0", "metrics/L1", "eval/accuracy"],
        aggregation="last"
    )

    print(f"\nDetailed info for run: {details['name']}")
    print(f"Run ID: {details['run_id']}")
    print(f"Config: {details['config']}")
    print(f"All Metrics: {details['metrics']}")

# Clean up - close all Neptune run connections
nq.cleanup()

print("\n" + "=" * 80)
print("Done! All Neptune connections closed.")
print("=" * 80)
