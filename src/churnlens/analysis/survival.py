"""Subscription spells, Kaplan–Meier curves and log-rank tests.

A spell is a run of non-cancel transactions where each starts within `gap_days`
of the running maximum expiry. A spell "churns" when its end + gap_days falls
before the cutoff (the gap is fully observed); otherwise it is right-censored at
the cutoff. Built from all sampled users' transactions before snapshot A's
cutoff. Spells first seen before `min_first_spell_start` are dropped because
they may continue a spell that began before the data starts (left truncation).
Cancellations are not used to shorten spells (documented limitation).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# "First transaction of a spell" key: earliest date, then every column as a tie-break,
# so same-day transactions resolve identically on every run.
FIRST_TXN_KEY = "seq"
# Total order over a user's non-cancel transactions: clean transactions are unique on all
# columns, and is_cancel is fixed at 0 here, so these columns never fully tie.
TOTAL_ORDER = ("transaction_date, membership_expire_date, actual_amount_paid, plan_list_price, "
               "payment_plan_days, is_auto_renew, payment_method_id")

# Grouping columns for KM curves: key -> (title, SQL label over the spells table).
GROUPINGS: dict[str, tuple[str, str]] = {
    "auto_renew": ("Auto-renew at spell start",
                   "CASE WHEN first_auto_renew = 1 THEN 'auto-renew' ELSE 'manual renew' END"),
    "first_payment": ("First payment of spell",
                      "CASE WHEN first_list_price = 0 THEN 'list price 0' "
                      "WHEN first_paid = 0 THEN 'free (paid 0)' "
                      "WHEN first_paid < first_list_price THEN 'discounted' "
                      "ELSE 'full price' END"),
    "plan_length": ("Plan length at spell start",
                    "CASE WHEN first_plan_days < 30 THEN 'under 30 days' "
                    "WHEN first_plan_days = 30 THEN '30 days' ELSE 'over 30 days' END"),
    "registered_via": ("Registration channel", "channel"),
}


def build_spells(con: duckdb.DuckDBPyConnection, cutoff: dt.date,
                 cfg: dict[str, Any], drop_left_truncated: bool = True) -> pd.DataFrame:
    """One row per subscription spell with duration (days), event flag and start attributes.

    drop_left_truncated=False keeps spells first seen before `min_first_spell_start`
    (needed when scoring every user's current spell; their age is then a lower bound).
    """
    gap = int(cfg["gap_days"])
    c = f"DATE '{cutoff.isoformat()}'"
    min_start = f"DATE '{cfg['min_first_spell_start']}'"
    keep = f"NOT (sp.first_seen AND sp.start < {min_start})" if drop_left_truncated else "true"
    return con.execute(f"""
        WITH t AS (
            -- seq: one total order per user, shared by both windows below. Windows that only
            -- ORDER BY (d, e) may break same-day ties differently and split a spell in two.
            SELECT msno, transaction_date AS d, membership_expire_date AS e, is_auto_renew,
                   actual_amount_paid, plan_list_price, payment_plan_days,
                   ROW_NUMBER() OVER (PARTITION BY msno ORDER BY {TOTAL_ORDER}) AS seq
            FROM transactions
            WHERE is_cancel = 0 AND transaction_date < {c} AND membership_expire_date IS NOT NULL
        ),
        o AS (
            SELECT *, MAX(e) OVER (PARTITION BY msno ORDER BY seq
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max
            FROM t
        ),
        s AS (
            SELECT *, SUM(CASE WHEN prev_max IS NULL OR d > prev_max + {gap} THEN 1 ELSE 0 END)
                          OVER (PARTITION BY msno ORDER BY seq ROWS UNBOUNDED PRECEDING)
                          AS spell_id,
                   prev_max IS NULL AS first_seen
            FROM o
        ),
        sp AS (
            SELECT msno, spell_id, MIN(d) AS start, MAX(e) AS spell_end,
                   BOOL_OR(first_seen) AS first_seen,
                   arg_min(is_auto_renew, {FIRST_TXN_KEY}) AS first_auto_renew,
                   arg_min(actual_amount_paid, {FIRST_TXN_KEY}) AS first_paid,
                   arg_min(plan_list_price, {FIRST_TXN_KEY}) AS first_list_price,
                   arg_min(payment_plan_days, {FIRST_TXN_KEY}) AS first_plan_days
            FROM s GROUP BY 1, 2
        )
        SELECT sp.*, COALESCE('channel ' || m.registered_via, 'no members row') AS channel,
               (sp.spell_end + {gap} < {c}) AS event,
               GREATEST(date_diff('day', sp.start, LEAST(sp.spell_end, {c})), 0) AS duration
        FROM sp LEFT JOIN members m USING (msno)
        WHERE {keep}""").df()


def km_by_group(con: duckdb.DuckDBPyConnection, spells: pd.DataFrame, key: str,
                cfg: dict[str, Any], alpha: float, top_n: int) -> dict[str, Any]:
    """KM curve per group (largest `top_n`), multivariate log-rank test, summary table."""
    title, expr = GROUPINGS[key]
    con.register("spells_df", spells)
    labels = con.execute(f"SELECT {expr} FROM spells_df").df().iloc[:, 0]
    con.unregister("spells_df")
    df = spells.assign(group=labels.values)
    largest = df["group"].value_counts().index[:top_n]
    df = df[df["group"].isin(largest)]
    test = multivariate_logrank_test(df["duration"], df["group"], df["event"])
    curves, rows = {}, []
    for group in largest:
        g = df[df["group"] == group]
        kmf = KaplanMeierFitter(alpha=alpha).fit(g["duration"], g["event"], label=str(group))
        curves[group] = kmf
        row = {"grouping": key, "group": group, "spells": len(g), "events": int(g["event"].sum()),
               "median_days": float(kmf.median_survival_time_)}
        for d in cfg["report_days"]:
            row[f"S({d}d)"] = float(kmf.predict(d))
        rows.append(row)
    return {"title": title, "curves": curves, "table": pd.DataFrame(rows),
            "logrank_stat": float(test.test_statistic), "logrank_p": float(test.p_value),
            "dof": int(df["group"].nunique() - 1)}
