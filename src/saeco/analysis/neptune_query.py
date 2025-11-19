"""
Neptune query tool for fetching and filtering runs based on metrics.

This implementation uses:
- neptune-query for efficient bulk data fetching (no run initialization)
- ProcessPoolExecutor for parallel processing when needed
- Caching to avoid re-downloading data

Example usage:
    from saeco.analysis.neptune_query import NeptuneQuery

    # Initialize query
    nq = NeptuneQuery("nqgl/default-project")

    # Find top 5 runs with lowest loss, subject to L0 < 50
    results = nq.query_topk(
        metric_key="loss",
        k=5,
        minimize=True,
        constraints={"L0": lambda x: x < 50},
        aggregation="last"  # or "mean_last_n" with n_steps=10
    )

    # Access run names and configs
    for run_info in results:
        print(f"Run: {run_info['name']}")
        print(f"Config: {run_info['config']}")
        print(f"Metrics: {run_info['metrics']}")
"""

from typing import Any, Callable, Literal, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from neptune_query import Query
import os


@dataclass
class RunInfo:
    """Information about a Neptune run."""

    run_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]

    def __repr__(self):
        return f"RunInfo(id={self.run_id}, name={self.name}, metrics={self.metrics})"


class NeptuneQuery:
    """
    Query Neptune runs with filtering and top-k selection.

    This class uses an optimized approach:
    1. Uses neptune-query for efficient bulk data fetching
    2. Never opens individual runs (pure read-only API queries)
    3. Optionally uses ProcessPoolExecutor for parallel fetching
    4. Caches query results to avoid re-downloading
    """

    def __init__(self, project: str, api_token: Optional[str] = None):
        """
        Initialize Neptune query.

        Args:
            project: Neptune project in format "workspace/project-name"
            api_token: Neptune API token (or set NEPTUNE_API_TOKEN env var)
        """
        self.project = project
        self.api_token = api_token or os.getenv("NEPTUNE_API_TOKEN")
        self._runs_cache: Optional[list[dict]] = None
        self._query_client = None

    def _init_query_client(self):
        """Initialize neptune-query client if not already done."""
        if self._query_client is None:
            self._query_client = Query(
                project=self.project,
                api_token=self.api_token
            )
        return self._query_client

    def fetch_runs(
        self,
        columns: Optional[list[str]] = None,
        force_refresh: bool = False,
        state: Optional[str] = None
    ) -> list[dict]:
        """
        Fetch all runs from the project efficiently using neptune-query.

        Args:
            columns: Specific columns/fields to fetch (e.g., ["sys/id", "sys/name", "train/loss"])
            force_refresh: Force re-fetching runs (ignore cache)
            state: Filter by run state ('active', 'inactive', etc.)

        Returns:
            List of run dictionaries with requested fields
        """
        if self._runs_cache is not None and not force_refresh:
            return self._runs_cache

        query = self._init_query_client()

        # Build query filter
        query_filter = {}
        if state:
            query_filter["sys/state"] = state

        # Fetch runs - neptune-query fetches everything efficiently
        if columns:
            runs_data = query.query(
                fields=columns,
                query=query_filter if query_filter else None
            )
        else:
            runs_data = query.query(query=query_filter if query_filter else None)

        # Convert to list of dicts
        runs_list = []
        for run in runs_data:
            runs_list.append(run)

        self._runs_cache = runs_list
        return runs_list

    def _get_metric_value(
        self,
        run_data: dict,
        metric_key: str,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10
    ) -> Optional[float]:
        """
        Extract metric value from run data.

        Args:
            run_data: Run data dictionary from neptune-query
            metric_key: Metric path (e.g., "train/loss" or "metrics/L0")
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation

        Returns:
            Metric value or None if not found
        """
        try:
            # neptune-query returns metrics directly
            # For "last" aggregation, the metric is typically available directly
            if metric_key in run_data:
                metric_data = run_data[metric_key]

                # Handle different data types
                if isinstance(metric_data, (int, float)):
                    return float(metric_data)
                elif isinstance(metric_data, dict):
                    # If it's a dict, try to get 'last' or 'value'
                    if 'last' in metric_data:
                        return float(metric_data['last'])
                    elif 'value' in metric_data:
                        return float(metric_data['value'])
                elif isinstance(metric_data, list):
                    # If it's a list of values, apply aggregation
                    values = np.array([float(v) for v in metric_data])
                    if len(values) == 0:
                        return None

                    if aggregation == "last":
                        return float(values[-1])
                    elif aggregation == "mean_last_n":
                        return float(np.mean(values[-n_steps:]))
                    elif aggregation == "min":
                        return float(np.min(values))
                    elif aggregation == "max":
                        return float(np.max(values))
                    elif aggregation == "mean_all":
                        return float(np.mean(values))

            return None

        except Exception as e:
            return None

    def _fetch_run_data(
        self,
        run_id: str,
        metric_keys: list[str]
    ) -> Optional[dict]:
        """
        Fetch specific metrics for a single run.

        Args:
            run_id: Neptune run ID
            metric_keys: List of metric keys to fetch

        Returns:
            Dict with run data or None
        """
        try:
            query = self._init_query_client()

            # Fetch specific run with specific fields
            fields = ["sys/id", "sys/name", "sys/state"] + metric_keys
            run_data = query.query_single_run(
                run_id=run_id,
                fields=fields
            )

            return run_data

        except Exception as e:
            return None

    def _fetch_runs_parallel(
        self,
        run_ids: list[str],
        metric_keys: list[str],
        max_workers: Optional[int] = None
    ) -> dict[str, dict]:
        """
        Fetch multiple runs in parallel using ProcessPoolExecutor.

        Args:
            run_ids: List of run IDs to fetch
            metric_keys: List of metric keys to fetch for each run
            max_workers: Maximum number of parallel workers (None = auto)

        Returns:
            Dict mapping run_id to run data
        """
        results = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_run_id = {
                executor.submit(self._fetch_run_data, run_id, metric_keys): run_id
                for run_id in run_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_run_id):
                run_id = future_to_run_id[future]
                try:
                    run_data = future.result()
                    if run_data:
                        results[run_id] = run_data
                except Exception as e:
                    # Skip failed runs
                    pass

        return results

    def query_topk(
        self,
        metric_key: str,
        k: int = 5,
        minimize: bool = True,
        constraints: Optional[dict[str, Callable[[float], bool]]] = None,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10,
        state: Optional[str] = None,
        use_parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> list[RunInfo]:
        """
        Query top-k runs based on a metric, subject to constraints.

        Args:
            metric_key: Metric to optimize (e.g., "train/loss")
            k: Number of top runs to return
            minimize: If True, return runs with lowest metric values
            constraints: Dict mapping metric keys to constraint functions
                        Example: {"L0": lambda x: x < 50, "L1": lambda x: x > 0.1}
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation
            state: Filter by run state
            use_parallel: Use parallel processing for fetching (faster for many runs)
            max_workers: Maximum parallel workers (None = auto)

        Returns:
            List of top-k RunInfo objects sorted by metric value

        Example:
            >>> nq = NeptuneQuery("myworkspace/myproject")
            >>> results = nq.query_topk(
            ...     metric_key="train/loss",
            ...     k=5,
            ...     minimize=True,
            ...     constraints={"metrics/L0": lambda x: x < 50}
            ... )
        """
        constraints = constraints or {}

        # Collect all metric keys we need
        all_metric_keys = [metric_key] + list(constraints.keys())

        # Build columns list for fetching
        columns = ["sys/id", "sys/name", "sys/state", "sys/tags"] + all_metric_keys

        # Fetch all runs with the needed fields
        runs_data = self.fetch_runs(columns=columns, state=state)

        # If using parallel fetching and we have many runs, fetch in parallel
        if use_parallel and len(runs_data) > 10:
            run_ids = [run["sys/id"] for run in runs_data]
            runs_detailed = self._fetch_runs_parallel(run_ids, all_metric_keys, max_workers)
            # Merge with basic data
            for run in runs_data:
                run_id = run["sys/id"]
                if run_id in runs_detailed:
                    run.update(runs_detailed[run_id])

        # Process each run
        valid_runs = []
        for run in runs_data:
            run_id = run.get("sys/id", "unknown")
            run_name = run.get("sys/name", run_id)

            # Extract metrics for this run
            run_metrics = {}

            # Check constraints first
            satisfies_constraints = True
            for constraint_key, constraint_fn in constraints.items():
                value = self._get_metric_value(run, constraint_key, aggregation, n_steps)

                if value is None:
                    satisfies_constraints = False
                    break

                run_metrics[constraint_key] = value

                if not constraint_fn(value):
                    satisfies_constraints = False
                    break

            if not satisfies_constraints:
                continue

            # Get target metric value
            metric_value = self._get_metric_value(run, metric_key, aggregation, n_steps)

            if metric_value is None:
                continue

            run_metrics[metric_key] = metric_value

            # Extract config from tags and other fields
            config = {}
            if "sys/tags" in run:
                config['tags'] = run["sys/tags"]

            # Try to get config fields
            for key in run:
                if key.startswith("config/") or key.startswith("parameters/"):
                    config[key] = run[key]

            # Create RunInfo
            run_info = RunInfo(
                run_id=run_id,
                name=run_name,
                config=config,
                metrics=run_metrics
            )
            valid_runs.append(run_info)

        # Sort by metric value
        valid_runs.sort(key=lambda r: r.metrics[metric_key], reverse=not minimize)

        # Return top-k
        return valid_runs[:k]

    def get_run_details(
        self,
        run_info: RunInfo,
        metric_keys: Optional[list[str]] = None,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10
    ) -> dict[str, Any]:
        """
        Get detailed information about a run.

        Args:
            run_info: RunInfo object
            metric_keys: List of metric keys to fetch (None = already cached metrics)
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation

        Returns:
            Dict with run details including all requested metrics
        """
        details = {
            "run_id": run_info.run_id,
            "name": run_info.name,
            "config": run_info.config.copy(),
            "metrics": run_info.metrics.copy()
        }

        # Fetch additional metrics if requested
        if metric_keys:
            run_data = self._fetch_run_data(run_info.run_id, metric_keys)
            if run_data:
                for metric_key in metric_keys:
                    if metric_key not in details["metrics"]:
                        value = self._get_metric_value(run_data, metric_key, aggregation, n_steps)
                        if value is not None:
                            details["metrics"][metric_key] = value

        return details

    def cleanup(self):
        """Clear cache and cleanup resources."""
        self._runs_cache = None
        self._query_client = None


# Example usage
if __name__ == "__main__":
    # Initialize query
    nq = NeptuneQuery("nqgl/default-project")

    # Find top 5 runs with lowest loss, subject to L0 < 50
    results = nq.query_topk(
        metric_key="train/loss",
        k=5,
        minimize=True,
        constraints={"metrics/L0": lambda x: x < 50},
        aggregation="last",
        use_parallel=True  # Use parallel processing
    )

    print("Top 5 runs with lowest loss (L0 < 50):")
    for i, run_info in enumerate(results, 1):
        print(f"\n{i}. {run_info.name}")
        print(f"   ID: {run_info.run_id}")
        print(f"   Metrics: {run_info.metrics}")
        print(f"   Config: {run_info.config}")

    # Clean up
    nq.cleanup()
