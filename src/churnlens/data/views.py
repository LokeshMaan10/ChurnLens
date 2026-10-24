"""DuckDB views over the raw Parquet layer (one view per table in configs/data.yaml).

Queries run inside DuckDB against Parquet, so large tables are never loaded
into pandas. Phase 1 adds cleaned views on top of these raw ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from churnlens.config import resolve_path


def table_source(spec: dict[str, Any], data_cfg: dict[str, Any]) -> Path:
    """Path of a table's raw Parquet output (a file, or a month-partitioned directory)."""
    return resolve_path(data_cfg["paths"]["raw_dir"], data_cfg) / spec["output"]


def parquet_scan_sql(path: Path) -> str:
    """read_parquet(...) expression for a file or a hive-partitioned directory."""
    if path.is_dir():
        # The `month` partition column is derived from `date`; keep only real columns.
        return (f"(SELECT * EXCLUDE (month) FROM read_parquet('{path.as_posix()}/**/*.parquet', "
                f"hive_partitioning = true))")
    return f"read_parquet('{path.as_posix()}')"


def register_raw_views(
    con: duckdb.DuckDBPyConnection, data_cfg: dict[str, Any], include_scoring_only: bool = True,
    prefix: str = "",
) -> list[str]:
    """Create one view per raw table that exists on disk. Returns the view names (unprefixed)."""
    names: list[str] = []
    for name, spec in data_cfg["tables"].items():
        if spec.get("scoring_only") and not include_scoring_only:
            continue
        path = table_source(spec, data_cfg)
        if not path.exists():
            continue
        con.execute(f"CREATE OR REPLACE VIEW {prefix}{name} AS "
                    f"SELECT * FROM {parquet_scan_sql(path)}")
        names.append(name)
    return names


def register_parquet_views(con: duckdb.DuckDBPyConnection, sources: dict[str, Path]) -> None:
    """Create one view per {view name: Parquet file or partitioned directory}."""
    for name, path in sources.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {parquet_scan_sql(path)}")
