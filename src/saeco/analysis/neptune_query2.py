from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import neptune_query as nq
import neptune_query.runs as nq_runs
import os

nq.set_api_token(os.getenv("NEPTUNE_API_TOKEN"))
MetricAgg = Literal["last", "mean_last_n", "min", "max", "mean_all"]


@dataclass(slots=True)
class RunInfo:
    """Information about a Neptune run (without opening Run objects)."""

    run_id: str
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    state: str | None = None

    def __repr__(self) -> str:
        return (
            f"RunInfo(id={self.run_id!r}, name={self.name!r}, metrics={self.metrics})"
        )


class NeptuneQuery:
    """Query Neptune runs with filtering and top-k selection using neptune-query.

    This version:
      * never calls `neptune.init_project` or `neptune.init_run`
      * pulls metadata & metrics via the `neptune_query` Runs API
      * can be used fully offline once you've materialized the DataFrames locally
    """

    def __init__(
        self,
        project: str | None = None,
        constraints: str | None = None,  # kept for compatibility, currently unused
        api_token: str | None = None,
        runs_selector: str | Sequence[str] | Any | None = None,
    ) -> None:
        """
        Args:
            project:
                Neptune project in format "workspace/project-name".
                If None, NEPTUNE_PROJECT env var is used.
            constraints:
                Deprecated / currently unused. Previously an NQL string.
            api_token:
                Optional Neptune API token. If None, NEPTUNE_API_TOKEN env var is used.
            runs_selector:
                Optional base selector for runs, passed as the `runs` argument to
                `neptune_query.runs.*` functions (regex, list of IDs, or Filter).
        """
        if api_token is not None:
            nq.set_api_token(api_token)

        self.project = project
        self._runs_selector = runs_selector
        self._constraints_nql = constraints  # just kept around for now

        # Cached runs metadata table (index: run, columns: attributes)
        self._runs_df: pd.DataFrame | None = None
        # Which metric attributes we’ve already asked `fetch_runs_table()` for
        self._metric_columns: set[str] = set()
        # Cached coarse RunInfo list (without per-metric aggregates)
        self._runs_cache: list[RunInfo] | None = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _attributes_pattern(metric_keys: Collection[str]) -> str:
        """Build a regex for `attributes=` that grabs sys info, configs, and metrics.

        We always include:
          * sys/name, sys/tags, sys/state
          * everything under config/, parameters/, params/
        and then any explicit metric keys we care about.
        """
        parts: list[str] = [
            r"^sys/name$",
            r"^sys/tags$",
            r"^sys/state$",
            r"^config/",
            r"^parameters/",
            r"^params/",
        ]
        for key in metric_keys:
            # Exact match for that metric path
            escaped = key.replace("/", r"\/")
            parts.append(rf"^{escaped}$")
        # Neptune examples use "a | b | c" style patterns
        return " | ".join(parts)

    def _ensure_runs_df(
        self,
        required_metric_keys: Collection[str] = (),
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch (or re-fetch) the runs table via neptune-query.

        Returns a DataFrame indexed by run ID, with one row per run.
        """
        required_metric_keys = set(required_metric_keys)

        if (
            self._runs_df is not None
            and not force_refresh
            and required_metric_keys <= self._metric_columns
        ):
            return self._runs_df

        # Union of all metrics we've ever requested (for "last" aggregation)
        self._metric_columns |= required_metric_keys
        attr_pattern = self._attributes_pattern(self._metric_columns)

        df = nq_runs.fetch_runs_table(
            project=self.project,
            runs=self._runs_selector,
            # attributes=attr_pattern,
        )

        # For series attributes, fetch_runs_table returns last logged value and uses
        # a MultiIndex ["run", "step"]. We only care about per-run rows.
        if isinstance(df.index, pd.MultiIndex) and "step" in df.index.names:
            df = df.reset_index("step", drop=True)

        df.index.name = "run"
        self._runs_df = df
        # Any time we re-fetch, we invalidate the high-level RunInfo cache
        self._runs_cache = None
        return df

    @staticmethod
    def _build_config(row: pd.Series) -> dict[str, Any]:
        """Collect config-like attributes and tags into a flat dict."""
        config: dict[str, Any] = {}

        # Tags
        if "sys/tags" in row.index:
            tags = row["sys/tags"]
            if pd.notna(tags):
                config["tags"] = tags

        # Config-style attributes (flattened)
        for col, value in row.items():
            if not isinstance(col, str):
                continue
            if not (
                col.startswith("config/")
                or col.startswith("parameters/")
                or col.startswith("params/")
            ):
                continue

            if pd.notna(value):
                config[col] = value

        return config

    @staticmethod
    def _aggregate_series(
        values: pd.Series,
        *,
        aggregation: MetricAgg,
        n_steps: int,
    ) -> pd.Series:
        """Aggregate a `Series` with MultiIndex index (run, step) over the run axis."""
        grouped = values.groupby(level="run")

        match aggregation:
            case "last":
                return grouped.apply(lambda s: s.iloc[-1])
            case "mean_last_n":
                return grouped.apply(lambda s: s.tail(n_steps).mean())
            case "min":
                return grouped.min()
            case "max":
                return grouped.max()
            case "mean_all":
                return grouped.mean()
            case other:
                raise ValueError(f"Unknown aggregation: {other!r}")

    def _fetch_metric_aggregates(
        self,
        run_ids: Sequence[str],
        metric_keys: Collection[str],
        *,
        aggregation: MetricAgg,
        n_steps: int,
    ) -> dict[str, dict[str, float]]:
        """Batch-fetch metric series and reduce to one scalar per (run, metric)."""
        if not metric_keys:
            return {}

        metric_keys = list(metric_keys)
        # For mean_last_n, we only need the tail; for others we want full history.
        tail_limit = n_steps if aggregation == "mean_last_n" else None

        metrics_df = nq_runs.fetch_metrics(
            project=self.project,
            runs=list(run_ids),
            attributes=metric_keys,
            tail_limit=tail_limit,
        )

        if metrics_df.empty:
            return {key: {} for key in metric_keys}

        out: dict[str, dict[str, float]] = {}

        # metrics_df columns: MultiIndex [metric_key, "value", ...]
        top_level_cols = {
            col[0] for col in metrics_df.columns
        }  # set of metric paths present

        for metric in metric_keys:
            if metric not in top_level_cols:
                out[metric] = {}
                continue

            values = metrics_df[metric]["value"]
            agg_series = self._aggregate_series(
                values, aggregation=aggregation, n_steps=n_steps
            )

            out[metric] = {str(run_id): float(v) for run_id, v in agg_series.items()}

        return out

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fetch_runs(
        self,
        state: str | None = None,
        force_refresh: bool = False,
    ) -> list[RunInfo]:
        """Fetch basic info for runs (no metric aggregation)."""
        if self._runs_cache is None or force_refresh:
            df = self._ensure_runs_df(force_refresh=force_refresh)
            runs: list[RunInfo] = []

            for run_id, row in df.iterrows():
                rid = str(run_id)
                runs.append(
                    RunInfo(
                        run_id=rid,
                        name=row.get("sys/name", rid),
                        config=self._build_config(row),
                        metrics={},
                        state=row.get("sys/state"),
                    )
                )

            self._runs_cache = runs

        if state is None:
            return list(self._runs_cache)

        return [r for r in self._runs_cache if r.state == state]

    def query_topk(
        self,
        metric_key: str,
        k: int = 5,
        *,
        minimize: bool = True,
        constraints: Mapping[str, Callable[[float], bool]] | None = None,
        aggregation: MetricAgg = "last",
        n_steps: int = 10,
        state: str | None = None,
    ) -> list[RunInfo]:
        """
        Query top-k runs based on a metric, subject to constraints.

        Args:
            metric_key:
                Metric to optimize (e.g., "train/loss" or "cache/L2_loss_").
            k:
                Number of top runs to return.
            minimize:
                If True, pick runs with *smallest* metric values. If False, largest.
            constraints:
                Mapping metric_key -> predicate on the aggregated value.
                Example: {"cache/L0_": lambda x: x < 60}
            aggregation:
                How to aggregate metric values over steps.
            n_steps:
                Used only when aggregation == "mean_last_n".
            state:
                Optional filter on `sys/state` ("active" / "inactive").
        """
        constraints = dict(constraints or {})
        all_metric_keys = {metric_key, *constraints.keys()}

        # For "last", we can get values directly from fetch_runs_table(). For other
        # aggregations, we still use fetch_runs_table() only for metadata/config,
        # and fetch_metrics() for the heavy lifting.
        runs_df = (
            self._ensure_runs_df(required_metric_keys=all_metric_keys)
            if aggregation == "last"
            else self._ensure_runs_df()
        )

        if "sys/state" in runs_df.columns and state is not None:
            runs_df = runs_df.loc[runs_df["sys/state"] == state]

        if runs_df.empty:
            return []

        run_ids = [str(r) for r in runs_df.index]

        # ------------------------------------------------------------------
        # Get aggregated values per (run, metric)
        # ------------------------------------------------------------------
        if aggregation == "last":
            metric_values: dict[str, dict[str, float]] = {}
            for key in all_metric_keys:
                if key not in runs_df.columns:
                    metric_values[key] = {}
                    continue
                col = runs_df[key]
                metric_values[key] = {
                    str(rid): float(v) for rid, v in col.items() if pd.notna(v)
                }
        else:
            metric_values = self._fetch_metric_aggregates(
                run_ids, all_metric_keys, aggregation=aggregation, n_steps=n_steps
            )

        # ------------------------------------------------------------------
        # Filter by constraints and build RunInfo objects
        # ------------------------------------------------------------------
        valid_runs: list[RunInfo] = []

        for run_id, row in runs_df.iterrows():
            rid = str(run_id)

            # Check constraints
            ok = True
            for key, fn in constraints.items():
                value = metric_values.get(key, {}).get(rid)
                if value is None or not fn(value):
                    ok = False
                    break
            if not ok:
                continue

            target_value = metric_values.get(metric_key, {}).get(rid)
            if target_value is None:
                continue

            metrics_for_run: dict[str, float] = {metric_key: target_value}
            for key in constraints:
                val = metric_values.get(key, {}).get(rid)
                if val is not None:
                    metrics_for_run[key] = val

            run_info = RunInfo(
                run_id=rid,
                name=row.get("sys/name", rid),
                config=self._build_config(row),
                metrics=metrics_for_run,
                state=row.get("sys/state"),
            )
            valid_runs.append(run_info)

        # Sort and truncate
        valid_runs.sort(key=lambda r: r.metrics[metric_key], reverse=not minimize)
        return valid_runs[:k]

    def get_run_details(
        self,
        run_info: RunInfo,
        metric_keys: list[str] | None = None,
        *,
        aggregation: MetricAgg = "last",
        n_steps: int = 10,
    ) -> dict[str, Any]:
        """
        Get detailed information about a run, optionally adding extra metrics.

        This does *not* open a Neptune Run; it uses `neptune_query.runs.fetch_metrics`.
        """
        details: dict[str, Any] = {
            "run_id": run_info.run_id,
            "name": run_info.name,
            "config": dict(run_info.config),
            "metrics": dict(run_info.metrics),
            "state": run_info.state,
        }

        if not metric_keys:
            return details

        missing = [m for m in metric_keys if m not in details["metrics"]]
        if not missing:
            return details

        agg = self._fetch_metric_aggregates(
            [run_info.run_id],
            missing,
            aggregation=aggregation,
            n_steps=n_steps,
        )
        for key in missing:
            value = agg.get(key, {}).get(run_info.run_id)
            if value is not None:
                details["metrics"][key] = value

        return details

    def export_runs_table(self, path: str) -> None:
        """Write the cached runs table to disk (e.g. Parquet/CSV) for offline work."""
        df = self._ensure_runs_df()
        if path.endswith(".parquet"):
            df.to_parquet(path)
        elif path.endswith(".csv"):
            df.to_csv(path)
        else:
            raise ValueError("Use a .parquet or .csv filename")

    def cleanup(self) -> None:
        """Drop all cached DataFrames / RunInfos (no remote side effects)."""
        self._runs_df = None
        self._runs_cache = None
        self._metric_columns.clear()


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize query (NEPTUNE_API_TOKEN + NEPTUNE_PROJECT env vars also work)
    nq_client = NeptuneQuery(
        project="nqgl/default-project",
        api_token=None,
    )

    results = nq_client.query_topk(
        metric_key="cache/L2_loss_",
        k=5,
        minimize=True,
        constraints={"cache/L0_": lambda x: x < 60.0},
        aggregation="last",
    )

    print("Top 5 runs with lowest loss (L0 < 60):")
    for i, run in enumerate(results, 1):
        print(f"\n{i}. {run.name} (id={run.run_id})")
        print(f"   Metrics: {run.metrics}")
        print(f"   Config keys: {list(run.config)[:10]}")

    # Optionally: materialize the runs table for offline querying
    # nq_client.export_runs_table("neptune_runs.parquet")
