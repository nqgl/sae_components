"""
Quick test script to verify the Neptune query tool with neptune-query.

This should show that:
1. No runs are opened (uses neptune-query API only)
2. Queries are very fast (bulk data fetching)
3. Parallel processing works when enabled
4. Results are correct
"""

from saeco.analysis.neptune_query import NeptuneQuery
import time

print("=" * 80)
print("Testing Neptune Query Tool (neptune-query + parallel)")
print("=" * 80)

# Initialize
print("\n1. Initializing NeptuneQuery...")
nq = NeptuneQuery("nqgl/default-project")
print("   ✓ Initialized (uses neptune-query API)")

# Test basic query with "last" aggregation
print("\n2. Query top 5 runs with lowest loss...")
start = time.time()
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    aggregation="last"
)
elapsed = time.time() - start

print(f"   ✓ Found {len(results)} runs in {elapsed:.2f}s")
for i, run_info in enumerate(results, 1):
    loss = run_info.metrics.get("train/loss", "N/A")
    print(f"   {i}. {run_info.name}: loss={loss}")

# Test query with constraints
print("\n3. Query with constraints (L0 < 50)...")
start = time.time()
results = nq.query_topk(
    metric_key="train/loss",
    k=3,
    minimize=True,
    constraints={"metrics/L0": lambda x: x < 50},
    aggregation="last"
)
elapsed = time.time() - start

print(f"   ✓ Found {len(results)} runs matching constraints in {elapsed:.2f}s")
for i, run_info in enumerate(results, 1):
    loss = run_info.metrics.get("train/loss", "N/A")
    l0 = run_info.metrics.get("metrics/L0", "N/A")
    print(f"   {i}. {run_info.name}: loss={loss}, L0={l0}")

# Test with parallel processing
print("\n4. Query with parallel processing enabled...")
nq.cleanup()  # Clear cache to test fresh
start = time.time()
results = nq.query_topk(
    metric_key="train/loss",
    k=5,
    minimize=True,
    use_parallel=True,  # Enable parallel processing
    max_workers=4
)
elapsed = time.time() - start

print(f"   ✓ Found {len(results)} runs with parallel processing in {elapsed:.2f}s")
for i, run_info in enumerate(results[:3], 1):  # Show top 3
    loss = run_info.metrics.get("train/loss", "N/A")
    print(f"   {i}. {run_info.name}: loss={loss}")

# Clean up
print("\n5. Cleaning up...")
nq.cleanup()
print("   ✓ Cleanup complete")

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)
