"""Cohort retention by registration month and channel (before snapshot A's cutoff).

Population: sampled members registered in the configured month range who have at
least one non-cancel transaction before the cutoff. "Active at age k" = some
non-cancel transaction made on/before the end of month (cohort + k) whose expiry is
on/after that month end. Cancellations are not netted out (documented limitation).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd


def month_start(value: Any) -> dt.date:
    """'YYYY-MM' (or a date) -> first day of that month."""
    if isinstance(value, dt.date):
        return value.replace(day=1)
    year, month = str(value).split("-")[:2]
    return dt.date(int(year), int(month), 1)


def cohort_activity(con: duckdb.DuckDBPyConnection, cutoff: dt.date,
                    cfg: dict[str, Any]) -> pd.DataFrame:
    """Rows: cohort month, channel, age k, users, active users. Only month ends before cutoff."""
    first = month_start(cfg["first_month"])
    last = month_start(cfg["last_month"])
    c = f"DATE '{cutoff.isoformat()}'"
    return con.execute(f"""
        WITH t AS (
            SELECT msno, transaction_date AS d, membership_expire_date AS e FROM transactions
            WHERE is_cancel = 0 AND transaction_date < {c} AND membership_expire_date IS NOT NULL
        ),
        base AS (
            SELECT m.msno, m.registered_via,
                   date_trunc('month', m.registration_date)::DATE AS cohort
            FROM members m
            WHERE m.registration_date >= DATE '{first.isoformat()}'
              AND m.registration_date < DATE '{last.isoformat()}' + INTERVAL 1 MONTH
              AND EXISTS (SELECT 1 FROM t WHERE t.msno = m.msno)
        ),
        grid AS (
            SELECT b.*, k.range AS k,
                   (b.cohort + to_months(k.range + 1) - INTERVAL 1 DAY)::DATE AS month_end
            FROM base b CROSS JOIN range(0, {int(cfg['max_age_months']) + 1}) k
        ),
        flagged AS (
            SELECT g.*, EXISTS (SELECT 1 FROM t WHERE t.msno = g.msno AND t.d <= g.month_end
                                AND t.e >= g.month_end) AS active
            FROM grid g WHERE g.month_end < {c}
        )
        SELECT cohort, registered_via, k, COUNT(*) AS users, SUM(active::INT) AS active
        FROM flagged GROUP BY ALL ORDER BY ALL""").df()


def cohort_matrix(activity: pd.DataFrame) -> pd.DataFrame:
    """Cohort month x age k -> active share (all channels)."""
    agg = activity.groupby(["cohort", "k"], as_index=False)[["users", "active"]].sum()
    agg["share"] = agg["active"] / agg["users"]
    return agg.pivot(index="cohort", columns="k", values="share")


def channel_curves(activity: pd.DataFrame, top: int) -> pd.DataFrame:
    """Active share by age k for the `top` largest channels (pooled over cohorts)."""
    sizes = activity[activity["k"] == 0].groupby("registered_via")["users"].sum()
    channels = sizes.sort_values(ascending=False).index[:top]
    sub = activity[activity["registered_via"].isin(channels)]
    agg = sub.groupby(["registered_via", "k"], as_index=False)[["users", "active"]].sum()
    agg["share"] = agg["active"] / agg["users"]
    agg["cohort_users"] = agg["registered_via"].map(sizes)
    return agg
