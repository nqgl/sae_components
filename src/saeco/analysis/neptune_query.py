"""
Neptune query tool for fetching and filtering runs based on metrics.

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
import neptune
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class RunInfo:
    """Information about a Neptune run."""

    run_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    _run_id_for_lazy_load: str = field(repr=False, default=None)
    _project: str = field(repr=False, default=None)
    _api_token: Optional[str] = field(repr=False, default=None)

    def __repr__(self):
        return f"RunInfo(id={self.run_id}, name={self.name}, metrics={self.metrics})"

    def _get_run_readonly(self):
        """Lazily open run in read-only mode only when needed."""
        if self._run_id_for_lazy_load and self._project:
            return neptune.init_run(
                project=self._project,
                api_token=self._api_token,
                with_id=self._run_id_for_lazy_load,
                mode="read-only"
            )
        return None


class NeptuneQuery:
    """
    Query Neptune runs with filtering and top-k selection.

    This class uses an optimized approach:
    1. Fetches runs table in bulk (fast, read-only)
    2. Gets metrics from table when possible (no run initialization)
    3. Only opens individual runs in read-only mode when needed for non-standard aggregations
    4. Avoids creating write-mode connections entirely
    """

    def __init__(self, project: str, api_token: Optional[str] = None):
        """
        Initialize Neptune query.

        Args:
            project: Neptune project in format "workspace/project-name"
            api_token: Neptune API token (or set NEPTUNE_API_TOKEN env var)
        """
        self.project = project
        self.api_token = api_token
        self._runs_table_cache: Optional[pd.DataFrame] = None
        self._project_handle = None

    def _init_project(self):
        """Initialize project handle if not already done."""
        if self._project_handle is None:
            self._project_handle = neptune.init_project(
                project=self.project,
                api_token=self.api_token,
                mode="read-only"
            )
        return self._project_handle

    def fetch_runs_table(
        self,
        state: Optional[str] = None,
        force_refresh: bool = False,
        columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch runs table from the project efficiently.

        Args:
            state: Filter by run state ('active', 'inactive', etc.)
            force_refresh: Force re-fetching runs (ignore cache)
            columns: Specific columns to fetch (e.g., ["sys/id", "sys/name", "train/loss"])

        Returns:
            Pandas DataFrame with runs data
        """
        if self._runs_table_cache is not None and not force_refresh:
            return self._runs_table_cache

        project = self._init_project()

        # Fetch runs table - this is MUCH more efficient than opening individual runs
        if columns:
            runs_table = project.fetch_runs_table(state=state, columns=columns).to_pandas()
        else:
            runs_table = project.fetch_runs_table(state=state).to_pandas()

        self._runs_table_cache = runs_table
        return runs_table

    def _get_metric_from_table_or_run(
        self,
        run_id: str,
        metric_key: str,
        runs_table: pd.DataFrame,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10
    ) -> Optional[float]:
        """
        Extract metric value from table if available, otherwise fetch from run.

        Args:
            run_id: Neptune run ID
            metric_key: Metric path (e.g., "train/loss_" or "metrics/L0_")
            runs_table: DataFrame with runs data
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation

        Returns:
            Metric value or None if not found
        """
        # First try to get from the table (most efficient)
        row = runs_table[runs_table['sys/id'] == run_id]
        if len(row) == 0:
            return None

        row = row.iloc[0]

        # Neptune stores metrics with trailing underscore in the table for last value
        table_key = metric_key if metric_key.endswith('_') else f"{metric_key}_"

        # For "last" aggregation, try to get directly from table
        if aggregation == "last" and table_key in runs_table.columns:
            value = row.get(table_key)
            if pd.notna(value):
                return float(value)

        # For other aggregations or if not in table, need to fetch from run
        # Only do this if absolutely necessary
        if aggregation != "last" or table_key not in runs_table.columns:
            return self._fetch_metric_from_run(run_id, metric_key, aggregation, n_steps)

        return None

    def _fetch_metric_from_run(
        self,
        run_id: str,
        metric_key: str,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"],
        n_steps: int
    ) -> Optional[float]:
        """
        Fetch metric from individual run (slower, only when needed).

        Args:
            run_id: Neptune run ID
            metric_key: Metric path
            aggregation: How to aggregate
            n_steps: Number of steps for mean_last_n

        Returns:
            Metric value or None
        """
        try:
            # Open run in read-only mode
            run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                with_id=run_id,
                mode="read-only"
            )

            try:
                # Fetch metric values
                metric_series = run[metric_key].fetch_values()

                if metric_series is None or len(metric_series) == 0:
                    return None

                # Convert to numpy array
                values = np.array([v.value for v in metric_series['value']])

                if len(values) == 0:
                    return None

                # Apply aggregation
                if aggregation == "last":
                    result = float(values[-1])
                elif aggregation == "mean_last_n":
                    result = float(np.mean(values[-n_steps:]))
                elif aggregation == "min":
                    result = float(np.min(values))
                elif aggregation == "max":
                    result = float(np.max(values))
                elif aggregation == "mean_all":
                    result = float(np.mean(values))
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

                return result
            finally:
                # Always close the run
                run.stop()

        except Exception as e:
            # Metric not found or error accessing it
            return None

    def query_topk(
        self,
        metric_key: str,
        k: int = 5,
        minimize: bool = True,
        constraints: Optional[dict[str, Callable[[float], bool]]] = None,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10,
        state: Optional[str] = None
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

        # Fetch runs table with specific columns for efficiency
        # Add trailing underscore for Neptune's metric naming convention
        columns = ["sys/id", "sys/name", "sys/tags", "sys/state"]
        for mk in all_metric_keys:
            table_key = mk if mk.endswith('_') else f"{mk}_"
            columns.append(table_key)

        runs_table = self.fetch_runs_table(state=state, columns=columns)

        # Process each run
        valid_runs = []
        for _, row in runs_table.iterrows():
            run_id = row['sys/id']
            run_name = row.get('sys/name', run_id)

            # Extract metrics for this run
            run_metrics = {}

            # Check constraints first
            satisfies_constraints = True
            for constraint_key, constraint_fn in constraints.items():
                value = self._get_metric_from_table_or_run(
                    run_id, constraint_key, runs_table, aggregation, n_steps
                )

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
            metric_value = self._get_metric_from_table_or_run(
                run_id, metric_key, runs_table, aggregation, n_steps
            )

            if metric_value is None:
                continue

            run_metrics[metric_key] = metric_value

            # Extract config from tags
            config = {}
            if 'sys/tags' in row and pd.notna(row['sys/tags']):
                config['tags'] = row['sys/tags']

            # Create RunInfo without opening the run
            run_info = RunInfo(
                run_id=run_id,
                name=run_name,
                config=config,
                metrics=run_metrics,
                _run_id_for_lazy_load=run_id,
                _project=self.project,
                _api_token=self.api_token
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
        n_steps: int = 10,
        fetch_full_config: bool = False
    ) -> dict[str, Any]:
        """
        Get detailed information about a run.

        Args:
            run_info: RunInfo object
            metric_keys: List of metric keys to fetch (None = already cached metrics)
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation
            fetch_full_config: If True, fetch full config from run (requires opening run)

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
            runs_table = self.fetch_runs_table()
            for metric_key in metric_keys:
                if metric_key not in details["metrics"]:
                    value = self._get_metric_from_table_or_run(
                        run_info.run_id,
                        metric_key,
                        runs_table,
                        aggregation=aggregation,
                        n_steps=n_steps
                    )
                    if value is not None:
                        details["metrics"][metric_key] = value

        # Fetch full config if requested (requires opening run)
        if fetch_full_config:
            run = run_info._get_run_readonly()
            if run:
                try:
                    for config_path in ['config', 'parameters', 'params']:
                        try:
                            config_data = run[config_path].fetch()
                            if isinstance(config_data, dict):
                                details["config"].update(config_data)
                        except:
                            pass
                finally:
                    run.stop()

        return details

    def cleanup(self):
        """Close project handle if open."""
        if self._project_handle:
            try:
                self._project_handle.stop()
            except:
                pass
            self._project_handle = None
        self._runs_table_cache = None


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
        aggregation="last"
    )

    print("Top 5 runs with lowest loss (L0 < 50):")
    for i, run_info in enumerate(results, 1):
        print(f"\n{i}. {run_info.name}")
        print(f"   ID: {run_info.run_id}")
        print(f"   Metrics: {run_info.metrics}")
        print(f"   Config: {run_info.config}")

    # Clean up
    nq.cleanup()
