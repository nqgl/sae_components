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
from dataclasses import dataclass, field


@dataclass
class RunInfo:
    """Information about a Neptune run."""

    run_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    raw_run: Any = field(repr=False)

    def __repr__(self):
        return f"RunInfo(id={self.run_id}, name={self.name}, metrics={self.metrics})"


class NeptuneQuery:
    """Query Neptune runs with filtering and top-k selection."""

    def __init__(self, project: str, api_token: Optional[str] = None):
        """
        Initialize Neptune query.

        Args:
            project: Neptune project in format "workspace/project-name"
            api_token: Neptune API token (or set NEPTUNE_API_TOKEN env var)
        """
        self.project = project
        self.api_token = api_token
        self._runs_cache: Optional[list[RunInfo]] = None

    def fetch_runs(self, state: Optional[str] = None, force_refresh: bool = False) -> list[RunInfo]:
        """
        Fetch all runs from the project.

        Args:
            state: Filter by run state ('active', 'inactive', etc.)
            force_refresh: Force re-fetching runs (ignore cache)

        Returns:
            List of RunInfo objects
        """
        if self._runs_cache is not None and not force_refresh:
            return self._runs_cache

        project = neptune.init_project(
            project=self.project,
            api_token=self.api_token,
            mode="read-only"
        )

        runs_table = project.fetch_runs_table(state=state).to_pandas()
        project.stop()

        run_infos = []
        for _, row in runs_table.iterrows():
            # Fetch detailed run info
            run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                with_id=row['sys/id'],
                mode="read-only"
            )

            # Extract config
            config = {}
            if 'sys/tags' in row:
                config['tags'] = row['sys/tags']

            # Try to get config from common config paths
            for config_path in ['config', 'parameters', 'params']:
                try:
                    config_data = run[config_path].fetch()
                    if isinstance(config_data, dict):
                        config.update(config_data)
                except:
                    pass

            # Store the run for later metric extraction
            run_info = RunInfo(
                run_id=row['sys/id'],
                name=row.get('sys/name', row['sys/id']),
                config=config,
                metrics={},
                raw_run=run
            )
            run_infos.append(run_info)

        self._runs_cache = run_infos
        return run_infos

    def _get_metric_value(
        self,
        run: Any,
        metric_key: str,
        aggregation: Literal["last", "mean_last_n", "min", "max", "mean_all"] = "last",
        n_steps: int = 10
    ) -> Optional[float]:
        """
        Extract metric value from a run.

        Args:
            run: Neptune run object
            metric_key: Metric path (e.g., "train/loss" or "metrics/L0")
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation

        Returns:
            Aggregated metric value or None if metric not found
        """
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
                return float(values[-1])
            elif aggregation == "mean_last_n":
                return float(np.mean(values[-n_steps:]))
            elif aggregation == "min":
                return float(np.min(values))
            elif aggregation == "max":
                return float(np.max(values))
            elif aggregation == "mean_all":
                return float(np.mean(values))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        except Exception as e:
            # Metric not found or error accessing it
            return None

    def _check_constraints(
        self,
        run_info: RunInfo,
        constraints: dict[str, Callable[[float], bool]],
        aggregation: str,
        n_steps: int
    ) -> bool:
        """
        Check if a run satisfies all constraints.

        Args:
            run_info: RunInfo object with raw_run
            constraints: Dict mapping metric keys to constraint functions
            aggregation: How to aggregate metric values
            n_steps: Number of steps for mean_last_n aggregation

        Returns:
            True if all constraints are satisfied
        """
        for metric_key, constraint_fn in constraints.items():
            value = self._get_metric_value(
                run_info.raw_run,
                metric_key,
                aggregation=aggregation,
                n_steps=n_steps
            )

            if value is None:
                return False

            if not constraint_fn(value):
                return False

        return True

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

        # Fetch all runs
        runs = self.fetch_runs(state=state)

        # Filter runs by constraints and extract target metric
        valid_runs = []
        for run_info in runs:
            # Check constraints
            if not self._check_constraints(run_info, constraints, aggregation, n_steps):
                continue

            # Get target metric value
            metric_value = self._get_metric_value(
                run_info.raw_run,
                metric_key,
                aggregation=aggregation,
                n_steps=n_steps
            )

            if metric_value is None:
                continue

            # Store metric in run_info
            run_info.metrics[metric_key] = metric_value

            # Also store constraint metrics for reference
            for constraint_key in constraints.keys():
                constraint_value = self._get_metric_value(
                    run_info.raw_run,
                    constraint_key,
                    aggregation=aggregation,
                    n_steps=n_steps
                )
                if constraint_value is not None:
                    run_info.metrics[constraint_key] = constraint_value

            valid_runs.append(run_info)

        # Sort by metric value
        valid_runs.sort(key=lambda r: r.metrics[metric_key], reverse=not minimize)

        # Return top-k
        result = valid_runs[:k]

        # Close runs that aren't in the result
        for run_info in runs:
            if run_info not in result and hasattr(run_info.raw_run, 'stop'):
                run_info.raw_run.stop()

        return result

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
            "config": run_info.config,
            "metrics": run_info.metrics.copy()
        }

        if metric_keys:
            for metric_key in metric_keys:
                if metric_key not in details["metrics"]:
                    value = self._get_metric_value(
                        run_info.raw_run,
                        metric_key,
                        aggregation=aggregation,
                        n_steps=n_steps
                    )
                    if value is not None:
                        details["metrics"][metric_key] = value

        return details

    def cleanup(self):
        """Close all cached runs."""
        if self._runs_cache:
            for run_info in self._runs_cache:
                if hasattr(run_info.raw_run, 'stop'):
                    try:
                        run_info.raw_run.stop()
                    except:
                        pass
            self._runs_cache = None


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
