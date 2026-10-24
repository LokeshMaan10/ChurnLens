"""Listening features: level, trend, consistency, quality (one row per label user).

A dense user x day grid over the longest window gives inactive days an explicit 0, so
slopes, variation and inactive-gap runs count silence correctly. Logs on/after the cutoff
are filtered again here (defence in depth).

Determinism: seconds stay exact DECIMALs through every sum (DuckDB turns DECIMAL / int
into DOUBLE, and float sums vary in the last bits between runs). Slope and coefficient
of variation are rebuilt from exact sums with a single final division.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

PLAYS = "num_25 + num_50 + num_75 + num_985 + num_100"


def _rate(num: str, den: str, cond: str) -> str:
    return (f"SUM({num}) FILTER (WHERE {cond})::DOUBLE / "
            f"NULLIF(SUM({den}) FILTER (WHERE {cond}), 0)")


def listening_sql(snap_dir: Path, cutoff: dt.date, cfg: dict[str, Any]) -> str:
    """SQL returning msno + every listening feature in the registry."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    windows = cfg["listening_windows_days"]
    longest, cap = max(windows), int(cfg["recency_cap_days"])
    cols: list[str] = []
    for w in windows:
        cols += [f"SUM(secs) FILTER (WHERE d <= {w}) / 60 AS minutes_{w}d",
                 f"COUNT(*) FILTER (WHERE active AND d <= {w}) AS active_days_{w}d",
                 f"COALESCE(SUM(plays) FILTER (WHERE d <= {w}), 0) AS plays_{w}d",
                 f"COALESCE(SUM(num_unq) FILTER (WHERE d <= {w}), 0) AS unique_songs_{w}d"]
    for p in cfg["ratio_pairs"]:
        r, b = p["recent"], p["base"]
        recent, base = f"d <= {r}", f"d > {r} AND d <= {r + b}"
        cols += [
            f"(SUM(secs) FILTER (WHERE {recent}) / {r}) / "
            f"NULLIF(SUM(secs) FILTER (WHERE {base}) / {b}, 0) "
            f"AS minutes_ratio_{r}_{b}",
            f"(COUNT(*) FILTER (WHERE active AND {recent}) / {r}) / "
            f"NULLIF(COUNT(*) FILTER (WHERE active AND {base}) / {b}, 0) "
            f"AS active_days_ratio_{r}_{b}"]
    for w in cfg["slope_windows_days"]:
        # OLS slope with x = -d over the w dense days: (nSxy - SxSy) / (nSxx - Sx^2),
        # numerator exact (DECIMAL x integer), seconds -> minutes at the end.
        f = f"FILTER (WHERE d <= {w})"
        cols.append(f"({w} * SUM(-d * secs) {f} - SUM(-d) {f} * SUM(secs) {f}) / "
                    f"({w} * SUM(d * d) {f} - SUM(-d) {f} * SUM(-d) {f}) / 60 "
                    f"AS minutes_slope_{w}d")
    cols += [
        # CV = sqrt(n*Syy - Sy^2) / Sy over n dense days (scale-free, so seconds work).
        f"sqrt(({longest} * SUM(secs * secs) - SUM(secs) * SUM(secs))::DOUBLE) / "
        f"NULLIF(SUM(secs), 0) AS minutes_cv_{longest}d",
        f"COALESCE(MIN(d) FILTER (WHERE active), {cap + 1}) AS days_since_last_listen",
        "SUM(secs) FILTER (WHERE d <= 30) / 60 / "
        "NULLIF(COUNT(*) FILTER (WHERE active AND d <= 30), 0) AS minutes_per_active_day_30d",
        f"COUNT(*) FILTER (WHERE secs_capped) AS capped_days_{longest}d",
    ]
    for q in cfg["quality_windows_days"]:
        cond = f"d <= {q}"
        cols += [f"{_rate('num_25', 'plays', cond)} AS skip_rate_{q}d",
                 f"{_rate('num_100', 'plays', cond)} AS completion_rate_{q}d",
                 f"{_rate('num_unq', 'plays', cond)} AS unique_song_share_{q}d"]
    recent, earlier = "d <= 30", f"d > 30 AND d <= {longest}"
    cols += [f"{_rate('num_25', 'plays', recent)} - {_rate('num_25', 'plays', earlier)} "
             "AS skip_rate_change_30_60",
             f"{_rate('num_100', 'plays', recent)} - {_rate('num_100', 'plays', earlier)} "
             "AS completion_rate_change_30_60"]
    select = ",\n               ".join(cols)
    return f"""
        WITH lab AS (SELECT msno FROM read_parquet('{s}/labels.parquet')),
        logs AS (
            SELECT msno, date_diff('day', date, {c}) AS d,
                   total_secs::DECIMAL(38, 6) AS secs, {PLAYS} AS plays,
                   num_25, num_100, num_unq, secs_capped
            FROM read_parquet('{s}/user_logs/**/*.parquet', hive_partitioning = true)
            WHERE date < {c} AND date >= {c} - INTERVAL {longest} DAY
        ),
        grid AS (
            SELECT l.msno, r.range AS d, COALESCE(g.secs, 0) AS secs,
                   g.msno IS NOT NULL AS active, g.plays, g.num_25, g.num_100, g.num_unq,
                   COALESCE(g.secs_capped, false) AS secs_capped
            FROM lab l CROSS JOIN range(1, {longest + 1}) r
            LEFT JOIN logs g ON g.msno = l.msno AND g.d = r.range
        ),
        runs AS (   -- consecutive inactive days: d - row_number is constant within a run
            SELECT msno, COUNT(*) AS run_len FROM (
                SELECT msno, d - ROW_NUMBER() OVER (PARTITION BY msno ORDER BY d) AS run_id
                FROM grid WHERE NOT active
            ) GROUP BY msno, run_id
        ),
        gaps AS (SELECT msno, MAX(run_len) AS longest FROM runs GROUP BY msno),
        agg AS (
            SELECT msno,
               {select}
            FROM grid GROUP BY msno
        )
        SELECT agg.*, COALESCE(gaps.longest, 0) AS longest_inactive_gap_{longest}d
        FROM agg LEFT JOIN gaps USING (msno)"""
