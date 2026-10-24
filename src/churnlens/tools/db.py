"""Read-only SQL access over curated views of the pipeline outputs.

Safety layers (each enforced in code, each tested):
  1. Exactly one statement, and its type must be in `allowed_statement_types` (SELECT).
  2. Table functions that read arbitrary files are rejected, so queries stay inside the
     curated views.
  3. A row cap (`max_rows`) and a timeout (connection interrupted after `timeout_seconds`).
  4. The connection is in-memory: there is no database file to write to.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import duckdb

from churnlens.config import load_config, resolve_path


class QueryRejected(ValueError):
    """The SQL was refused before execution (not read-only, or outside the views)."""


@dataclass
class QueryResult:
    """Rows plus the exact SQL that ran."""

    sql_executed: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {"sql_executed": self.sql_executed, "columns": self.columns, "rows": self.rows,
                "row_count": self.row_count, "truncated": self.truncated}


def _views(cfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {name: (resolve_path(v["source"], data_cfg).as_posix(), v["description"].strip())
            for name, v in cfg["views"].items()}


@lru_cache(maxsize=1)
def connection() -> duckdb.DuckDBPyConnection:
    """In-memory connection with one view per available curated source."""
    cfg, data_cfg = load_config("tools"), load_config("data")
    con = duckdb.connect()
    for name, (path, _) in _views(cfg, data_cfg).items():
        if resolve_path(path, data_cfg).exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def schema(table: str | None = None) -> dict[str, Any]:
    """Columns and descriptions of the curated views (all, or one)."""
    cfg, data_cfg = load_config("tools"), load_config("data")
    views = _views(cfg, data_cfg)
    con = connection()
    available = {r[0] for r in con.execute(
        "SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    names = [table] if table else sorted(available)
    out = {}
    for name in names:
        if name not in available:
            raise KeyError(f"unknown table {name!r}; available: {sorted(available)}")
        cols = con.execute(f"DESCRIBE {name}").fetchall()
        out[name] = {"description": views[name][1],
                     "columns": [{"name": c[0], "type": c[1]} for c in cols]}
    return out


def validate(sql: str, cfg: dict[str, Any]) -> str:
    """Return the single statement if it is allowed; raise QueryRejected otherwise."""
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1:
        raise QueryRejected(f"exactly one statement is allowed (got {len(statements)})")
    stmt = statements[0]
    kind = str(stmt.type).split(".")[-1]
    if kind not in cfg["allowed_statement_types"]:
        raise QueryRejected(f"only read-only {cfg['allowed_statement_types']} statements are "
                            f"allowed; this is a {kind} statement")
    text = stmt.query
    for fn in cfg["blocked_functions"]:
        if re.search(rf"\b{re.escape(fn)}\b", text, flags=re.IGNORECASE):
            raise QueryRejected(f"function {fn!r} is not allowed; query the curated views")
    return text.strip().rstrip(";")


def run_sql(sql: str, max_rows: int | None = None,
            timeout_seconds: float | None = None) -> QueryResult:
    """Run one read-only query over the curated views, with a row cap and a timeout."""
    cfg = load_config("tools")["sql"]
    limit = int(max_rows or cfg["max_rows"])
    timeout = float(timeout_seconds or cfg["timeout_seconds"])
    body = validate(sql, cfg)
    wrapped = f"SELECT * FROM ({body}) AS q LIMIT {limit + 1}"
    con = connection().cursor()
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        rel = con.execute(wrapped)
        cols = [d[0] for d in rel.description]
        rows = [list(r) for r in rel.fetchall()]
    except duckdb.InterruptException as e:
        raise TimeoutError(f"query exceeded {timeout:.0f}s and was stopped") from e
    finally:
        timer.cancel()
    truncated = len(rows) > limit
    rows = rows[:limit]
    return QueryResult(body, cols, [[_jsonable(v) for v in r] for r in rows], len(rows),
                       truncated)


def _jsonable(v: Any) -> Any:
    """Make DuckDB values JSON-friendly (dates/decimals/numpy -> str/float)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
