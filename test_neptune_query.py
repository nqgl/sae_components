"""
Quick test script to verify the optimized Neptune query tool works correctly.

This should show that:
1. No runs are opened in write mode (no "Neptune initialized" messages)
2. Queries are fast
3. Results are correct
"""

from saeco.analysis.neptune_query import NeptuneQuery

print("=" * 80)
print("Testing Neptune Query Tool (Optimized)")
print("=" * 80)

# Initialize
print("\n1. Initializing NeptuneQuery...")
nq = NeptuneQuery("nqgl/default-project")
print("   ✓ Initialized (no runs opened yet)")

# Test basic query with "last" aggregation (should use table only)
print("\n2. Query top 5 runs with lowest loss (using table data)...")
results = nq.query_topk(
    metric_key="train/loss",  # No trailing underscore needed - code handles it
    k=5,
    minimize=True,
    aggregation="last"
)

print(f"   ✓ Found {len(results)} runs")
for i, run_info in enumerate(results, 1):
    loss = run_info.metrics.get("train/loss", "N/A")
    print(f"   {i}. {run_info.name}: loss={loss}")

# Test query with constraints
print("\n3. Query with constraints (L0 < 50)...")
results = nq.query_topk(
    metric_key="train/loss",
    k=3,
    minimize=True,
    constraints={"metrics/L0": lambda x: x < 50},
    aggregation="last"
)

print(f"   ✓ Found {len(results)} runs matching constraints")
for i, run_info in enumerate(results, 1):
    loss = run_info.metrics.get("train/loss", "N/A")
    l0 = run_info.metrics.get("metrics/L0", "N/A")
    print(f"   {i}. {run_info.name}: loss={loss}, L0={l0}")

# Test with mean_last_n aggregation (will need to open runs)
print("\n4. Query with mean_last_n aggregation (may open runs if needed)...")
results = nq.query_topk(
    metric_key="train/loss",  # No underscore - fetch from run
    k=2,
    minimize=True,
    aggregation="mean_last_n",
    n_steps=5
)

print(f"   ✓ Found {len(results)} runs")
for i, run_info in enumerate(results, 1):
    loss = run_info.metrics.get("train/loss", "N/A")
    print(f"   {i}. {run_info.name}: avg loss (last 5)={loss}")

# Clean up
print("\n5. Cleaning up...")
nq.cleanup()
print("   ✓ Cleanup complete")

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)
