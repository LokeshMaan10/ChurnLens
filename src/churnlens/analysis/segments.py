"""Churn rate by customer segment, promotion analysis, and listening trajectories.

All inputs come from one snapshot's directory (data before its cutoff only).
Outputs are descriptive associations, never causal effects.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from churnlens.analysis.stats import Rate, chi2_independence, two_proportion_z, wilson
from churnlens.data.ordering import TXN_LATEST_FIRST

OTHER = "Other"
# Deterministic "latest plan" ordering (shared with feature code).
LAST_PLAN_ORDER = TXN_LATEST_FIRST


@dataclass
class Dimension:
    """A segmentation dimension: a SQL label expression over the `base` table."""

    key: str
    title: str
    expr: str
    ordered: bool = False   # natural order (bands) instead of size order
    fold: bool = True       # fold small/extra categories into "Other"


def band_case(column: str, edges: list[float], unit: str, null_label: str) -> str:
    """SQL CASE mapping a numeric column to left-closed band labels, sortable by prefix."""
    parts = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        label = f"{i:02d}|{lo:g}–{hi:g}{unit}"
        parts.append(f"WHEN {column} >= {lo} AND {column} < {hi} THEN '{label}'")
    return f"CASE WHEN {column} IS NULL THEN '99|{null_label}' {' '.join(parts)} " \
           f"ELSE '98|out of range' END"


def dimensions(cfg: dict[str, Any]) -> list[Dimension]:
    """The segment dimensions analysed in Phase 3."""
    return [
        Dimension("auto_renew", "Auto-renew (last plan)",
                  "CASE is_auto_renew WHEN 1 THEN 'auto-renew' WHEN 0 THEN 'manual renew' "
                  "ELSE 'no transaction' END"),
        Dimension("price_type", "Price paid vs list (last plan)", "price_type", fold=False),
        Dimension("plan_days", "Plan length, days (last plan)",
                  "COALESCE(CAST(payment_plan_days AS VARCHAR), 'no transaction')"),
        Dimension("list_price", "List price, NTD (last plan)",
                  "COALESCE(CAST(plan_list_price AS VARCHAR), 'no transaction')"),
        Dimension("payment_method", "Payment method (last plan)",
                  "COALESCE('method ' || payment_method_id, 'no transaction')"),
        Dimension("ever_cancelled", "Cancelled before cutoff",
                  "CASE WHEN ever_cancelled THEN 'has cancelled' ELSE 'never cancelled' END"),
        Dimension("registered_via", "Registration channel",
                  "COALESCE('channel ' || registered_via, 'no members row')"),
        Dimension("city", "City", "COALESCE('city ' || city, 'no members row')"),
        Dimension("gender", "Gender",
                  "CASE WHEN NOT has_member THEN 'no members row' "
                  "ELSE COALESCE(gender, 'unknown') END", fold=False),
        Dimension("age_band", "Age band",
                  band_case("age", cfg["age_band_edges"], "", "unknown / invalid"),
                  ordered=True, fold=False),
        Dimension("tenure_band", "Tenure at cutoff",
                  band_case("tenure_years", cfg["tenure_band_edges_years"], "y",
                            "unknown"), ordered=True, fold=False),
    ]


def build_base(con: duckdb.DuckDBPyConnection, snap_dir: Path, cutoff: dt.date) -> None:
    """One row per label user: label, last non-cancel pre-cutoff plan, member attributes."""
    s = snap_dir.as_posix()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE base AS
        WITH lab AS (SELECT * FROM read_parquet('{s}/labels.parquet')),
        tx AS (SELECT * FROM read_parquet('{s}/transactions.parquet')),
        mem AS (SELECT * FROM read_parquet('{s}/members.parquet')),
        last AS (
            -- Full tie-break so same-day transactions resolve identically on every run.
            SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY msno
                           ORDER BY {LAST_PLAN_ORDER}) AS rn
                           FROM tx WHERE is_cancel = 0) WHERE rn = 1
        ),
        canc AS (SELECT DISTINCT msno FROM tx WHERE is_cancel = 1)
        SELECT l.msno, l.is_churn, t.payment_plan_days, t.plan_list_price,
               t.actual_amount_paid, t.is_auto_renew, t.payment_method_id,
               CASE WHEN t.msno IS NULL THEN 'no transaction'
                    WHEN t.plan_list_price = 0 THEN 'list price 0'
                    WHEN t.actual_amount_paid = 0 THEN 'free (paid 0)'
                    WHEN t.actual_amount_paid < t.plan_list_price THEN 'discounted'
                    WHEN t.actual_amount_paid > t.plan_list_price THEN 'paid above list'
                    ELSE 'full price' END AS price_type,
               m.msno IS NOT NULL AS has_member, m.registered_via, m.city, m.age, m.gender,
               date_diff('day', m.registration_date, DATE '{cutoff.isoformat()}') / 365.25
                   AS tenure_years,
               c.msno IS NOT NULL AS ever_cancelled
        FROM lab l LEFT JOIN last t USING (msno) LEFT JOIN mem m USING (msno)
        LEFT JOIN canc c USING (msno)""")


def fold_groups(rows: list[tuple[str, int, int]], dim: Dimension, min_n: int,
                max_cats: int) -> list[tuple[str, int, int]]:
    """Fold small / surplus categories into Other; order by size or by band prefix."""
    if dim.ordered:
        rows = sorted(rows)
        return [(lab.split("|", 1)[1], k, n) for lab, k, n in rows]
    rows = sorted(rows, key=lambda r: -r[2])
    if not dim.fold:
        return rows
    keep, other_k, other_n = [], 0, 0
    for lab, k, n in rows:
        if n >= min_n and len(keep) < max_cats:
            keep.append((lab, k, n))
        else:
            other_k, other_n = other_k + k, other_n + n
    if other_n:
        keep.append((OTHER, other_k, other_n))
    return keep


def segment_rates(con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Churn rate with Wilson CI per group, and a chi-square test per dimension."""
    alpha = cfg["alpha"]
    group_rows, test_rows = [], []
    for dim in dimensions(cfg):
        raw = con.execute(f"SELECT {dim.expr} AS g, SUM(is_churn)::INT, COUNT(*) FROM base "
                          "GROUP BY 1").fetchall()
        raw = [(str(g), int(k), int(n)) for g, k, n in raw]
        chi2, p, v = chi2_independence([(k, n) for _, k, n in raw])
        groups = fold_groups(raw, dim, cfg["min_group_size"], cfg["max_categories"])
        for order, (label, k, n) in enumerate(groups):
            r = wilson(k, n, alpha)
            group_rows.append({"dimension": dim.key, "title": dim.title, "order": order,
                               "group": label, "n": n, "churners": k, "rate": r.rate,
                               "ci_lo": r.lo, "ci_hi": r.hi})
        test_rows.append({"dimension": dim.key, "title": dim.title, "groups_tested": len(raw),
                          "chi2": chi2, "p_value": p, "cramers_v": v})
    tests = pd.DataFrame(test_rows).sort_values("cramers_v", ascending=False)
    return pd.DataFrame(group_rows), tests


def promo_analysis(con: duckdb.DuckDBPyConnection, snap_dir: Path, cutoff: dt.date,
                   cfg: dict[str, Any]) -> pd.DataFrame:
    """Churn after promotional vs full-price periods (last plan, and recent history)."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    look = int(cfg["promo_lookback_days"])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE promo AS
        SELECT b.msno, b.is_churn, b.price_type,
               EXISTS (SELECT 1 FROM read_parquet('{s}/transactions.parquet') t
                       WHERE t.msno = b.msno AND t.is_cancel = 0
                       AND t.transaction_date >= {c} - INTERVAL {look} DAY
                       AND (t.plan_list_price = 0 OR t.actual_amount_paid < t.plan_list_price))
                   AS promo_recent
        FROM base b""")
    rows = []
    full = con.execute("SELECT SUM(is_churn)::INT, COUNT(*) FROM promo "
                       "WHERE price_type = 'full price'").fetchone()
    for label, where in (
        ("last plan: full price", "price_type = 'full price'"),
        ("last plan: discounted (0 < paid < list)", "price_type = 'discounted'"),
        ("last plan: free (paid 0, list > 0)", "price_type = 'free (paid 0)'"),
        ("last plan: list price 0", "price_type = 'list price 0'"),
        ("last plan: any promotional (3 rows above)",
         "price_type IN ('discounted', 'free (paid 0)', 'list price 0')"),
        (f"promo in last {look} days: yes", "promo_recent"),
        (f"promo in last {look} days: no", "NOT promo_recent AND price_type <> 'no transaction'"),
    ):
        k, n = con.execute(f"SELECT SUM(is_churn)::INT, COUNT(*) FROM promo WHERE {where}"
                           ).fetchone()
        k = k or 0
        r: Rate = wilson(k, n, cfg["alpha"])
        ref = full if label.startswith("last plan") else con.execute(
            "SELECT SUM(is_churn)::INT, COUNT(*) FROM promo WHERE NOT promo_recent "
            "AND price_type <> 'no transaction'").fetchone()
        z, p = two_proportion_z(k, n, ref[0], ref[1]) if label not in (
            "last plan: full price", f"promo in last {look} days: no") else (float("nan"),) * 2
        rows.append({"group": label, "n": n, "churners": k, "rate": r.rate, "ci_lo": r.lo,
                     "ci_hi": r.hi, "z_vs_reference": z, "p_value": p})
    return pd.DataFrame(rows)


def trajectories(con: duckdb.DuckDBPyConnection, snap_dir: Path, cutoff: dt.date,
                 days: int) -> pd.DataFrame:
    """Daily listening metrics per label group for the `days` days before the cutoff.

    Per-user means divide by the full group size, so a day without listening counts
    as zero. Skip and completion rates are pooled play shares within the group.
    """
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    df = con.execute(f"""
        WITH lab AS (SELECT * FROM read_parquet('{s}/labels.parquet')),
        sizes AS (SELECT is_churn, COUNT(*) AS group_n FROM lab GROUP BY 1),
        logs AS (
            SELECT g.*, l.is_churn FROM read_parquet('{s}/user_logs/**/*.parquet',
                hive_partitioning = true) g JOIN lab l USING (msno)
            WHERE g.date >= {c} - INTERVAL {days} DAY
        )
        SELECT date_diff('day', {c}, date) AS day, logs.is_churn, ANY_VALUE(group_n) AS group_n,
               COUNT(*) AS active_users,
               -- DECIMAL sums are exact, so results do not depend on parallel summation order.
               SUM(total_secs::DECIMAL(38, 6))::DOUBLE AS secs,
               SUM(num_25) AS n25, SUM(num_100) AS n100, SUM(num_unq) AS unq,
               SUM(num_25 + num_50 + num_75 + num_985 + num_100) AS plays
        FROM logs JOIN sizes USING (is_churn) GROUP BY 1, 2 ORDER BY 1, 2""").df()
    df["minutes_per_user"] = df["secs"] / 60 / df["group_n"]
    df["active_share"] = df["active_users"] / df["group_n"]
    df["skip_rate"] = df["n25"] / df["plays"]
    df["completion_rate"] = df["n100"] / df["plays"]
    df["unique_songs_per_user"] = df["unq"] / df["group_n"]
    return df


def calendar_check(con: duckdb.DuckDBPyConnection, cutoff: dt.date, recent: int,
                   baseline: int) -> dict[str, Any]:
    """All-user activity in the last `recent` days before a cutoff vs the `baseline` before it.

    Uses every sampled user's clean training logs (no labels), so it measures the
    calendar, not churn. Strictly pre-cutoff data only.
    """
    c = f"DATE '{cutoff.isoformat()}'"
    row = con.execute(f"""
        WITH d AS (
            SELECT date, COUNT(*) AS users,
                   SUM(total_secs::DECIMAL(38, 6))::DOUBLE / COUNT(*) / 60 AS min_per_active
            FROM user_logs
            WHERE date < {c} AND date >= {c} - INTERVAL {recent + baseline} DAY GROUP BY 1
        )
        SELECT AVG(users) FILTER (WHERE date >= {c} - INTERVAL {recent} DAY),
               AVG(users) FILTER (WHERE date < {c} - INTERVAL {recent} DAY),
               AVG(min_per_active) FILTER (WHERE date >= {c} - INTERVAL {recent} DAY),
               AVG(min_per_active) FILTER (WHERE date < {c} - INTERVAL {recent} DAY)
        FROM d""").fetchone()
    return {"cutoff": cutoff.isoformat(), "recent_active_users": row[0],
            "baseline_active_users": row[1], "active_change": row[0] / row[1] - 1,
            "recent_min_per_active": row[2], "baseline_min_per_active": row[3],
            "minutes_change": row[2] / row[3] - 1}


def trajectory_summary(con: duckdb.DuckDBPyConnection, snap_dir: Path, cutoff: dt.date,
                       days: int, recent: int) -> pd.DataFrame:
    """Per-user recent-vs-earlier listening, summarised by label group."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    earlier = days - recent
    return con.execute(f"""
        WITH lab AS (SELECT * FROM read_parquet('{s}/labels.parquet')),
        logs AS (SELECT * FROM read_parquet('{s}/user_logs/**/*.parquet',
                 hive_partitioning = true) WHERE date >= {c} - INTERVAL {days} DAY),
        per_user AS (
            SELECT l.msno, l.is_churn,
                   COALESCE(SUM(g.total_secs::DECIMAL(38, 6))
                            FILTER (WHERE g.date >= {c} - INTERVAL {recent} DAY),
                            0)::DOUBLE / 60 / {recent} AS recent_min_per_day,
                   COALESCE(SUM(g.total_secs::DECIMAL(38, 6))
                            FILTER (WHERE g.date < {c} - INTERVAL {recent} DAY),
                            0)::DOUBLE / 60 / {earlier} AS earlier_min_per_day,
                   COUNT(g.date) FILTER (WHERE g.date >= {c} - INTERVAL {recent} DAY)
                       AS recent_active_days
            FROM lab l LEFT JOIN logs g USING (msno) GROUP BY 1, 2
        )
        SELECT is_churn, COUNT(*) AS users,
               MEDIAN(recent_min_per_day) AS median_recent_min_per_day,
               MEDIAN(earlier_min_per_day) AS median_earlier_min_per_day,
               MEDIAN(recent_min_per_day / NULLIF(earlier_min_per_day, 0))
                   AS median_recent_to_earlier_ratio,
               AVG((recent_min_per_day < 0.5 * earlier_min_per_day)::INT)
                   AS share_down_over_half,
               MEDIAN(recent_active_days) AS median_recent_active_days,
               AVG((recent_active_days = 0)::INT) AS share_inactive_recent
        FROM per_user GROUP BY 1 ORDER BY 1""").df()
