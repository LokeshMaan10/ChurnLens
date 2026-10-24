"""Plan, price, discount, renewal, cancellation and payment features.

Only transactions dated strictly before the cutoff are used (re-filtered here as a
second guard), so renewals and cancellations inside the label window never leak in.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from churnlens.data.ordering import TXN_LATEST_FIRST, TXN_OLDEST_FIRST


def transactions_sql(snap_dir: Path, cutoff: dt.date, cfg: dict[str, Any]) -> str:
    """SQL returning msno + every transaction feature in the registry."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    lb, pb = int(cfg["transaction_lookback_days"]), int(cfg["promo_lookback_days"])
    in_lb = f"transaction_date >= {c} - INTERVAL {lb} DAY"
    return f"""
        WITH lab AS (SELECT msno FROM read_parquet('{s}/labels.parquet')),
        tx AS (SELECT * FROM read_parquet('{s}/transactions.parquet')
               WHERE transaction_date < {c}),
        nc AS (SELECT * FROM tx WHERE is_cancel = 0),
        last AS (
            SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY msno
                           ORDER BY {TXN_LATEST_FIRST}) AS rn FROM nc) WHERE rn = 1
        ),
        latest_any AS (   -- most recent transaction of any kind; same-day cancel ranks last
            SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY msno
                           ORDER BY transaction_date DESC, is_cancel DESC, {TXN_LATEST_FIRST})
                           AS rn FROM tx) WHERE rn = 1
        ),
        seq AS (
            SELECT msno, payment_plan_days, plan_list_price, payment_method_id,
                   LAG(payment_plan_days) OVER w AS prev_days,
                   LAG(plan_list_price) OVER w AS prev_price,
                   LAG(payment_method_id) OVER w AS prev_method
            FROM nc WHERE {in_lb}
            WINDOW w AS (PARTITION BY msno ORDER BY {TXN_OLDEST_FIRST})
        ),
        changes AS (
            SELECT msno,
                   COUNT(*) FILTER (WHERE prev_days IS NOT NULL AND
                       (payment_plan_days <> prev_days OR plan_list_price <> prev_price))
                       AS plan_changes,
                   COUNT(*) FILTER (WHERE prev_method IS NOT NULL AND
                       payment_method_id <> prev_method) AS method_changes
            FROM seq GROUP BY msno
        ),
        agg AS (
            SELECT msno,
                   COUNT(*) FILTER (WHERE is_cancel = 0 AND {in_lb}) AS n_tx,
                   SUM(actual_amount_paid) FILTER (WHERE is_cancel = 0 AND {in_lb}) AS paid,
                   AVG(is_auto_renew) FILTER (WHERE is_cancel = 0 AND {in_lb}) AS share_ar,
                   COUNT(*) FILTER (WHERE is_cancel = 0
                       AND transaction_date >= {c} - INTERVAL {pb} DAY
                       AND (plan_list_price = 0 OR actual_amount_paid < plan_list_price))
                       AS n_promo,
                   COUNT(*) FILTER (WHERE is_cancel = 1) AS n_cancel_all,
                   COUNT(*) FILTER (WHERE is_cancel = 1 AND {in_lb}) AS n_cancel_lb,
                   MAX(transaction_date) FILTER (WHERE is_cancel = 1) AS last_cancel,
                   MAX(transaction_date) AS last_tx, MIN(transaction_date) AS first_tx,
                   MAX(membership_expire_date) FILTER (WHERE is_cancel = 0) AS max_expiry
            FROM tx GROUP BY msno
        )
        SELECT l.msno,
               last.payment_plan_days AS last_plan_days,
               last.plan_list_price AS last_list_price,
               last.actual_amount_paid AS last_amount_paid,
               last.plan_list_price - last.actual_amount_paid AS last_discount_amount,
               last.actual_amount_paid::DOUBLE / NULLIF(last.payment_plan_days, 0)
                   AS last_price_per_day,
               (last.actual_amount_paid > 0 AND last.actual_amount_paid < last.plan_list_price)
                   ::INT AS last_is_discounted,
               (last.actual_amount_paid = 0)::INT AS last_is_free,
               (last.plan_list_price = 0)::INT AS last_list_price_zero,
               last.is_auto_renew::INT AS last_is_auto_renew,
               CAST(last.payment_method_id AS VARCHAR) AS last_payment_method,
               latest_any.is_cancel::INT AS last_txn_is_cancel,
               date_diff('day', {c}, agg.max_expiry) AS days_to_expiry,
               date_diff('day', {c}, latest_any.membership_expire_date)
                   AS days_to_latest_txn_expiry,
               COALESCE(agg.n_tx, 0) AS n_transactions_{lb}d,
               COALESCE(agg.paid, 0) AS total_paid_{lb}d,
               agg.share_ar::DOUBLE AS share_auto_renew_{lb}d,
               COALESCE(changes.plan_changes, 0) AS n_plan_changes_{lb}d,
               COALESCE(changes.method_changes, 0) AS n_payment_method_changes_{lb}d,
               COALESCE(agg.n_promo, 0) AS n_promo_txns_{pb}d,
               COALESCE(agg.n_cancel_all, 0) AS n_cancels_all,
               COALESCE(agg.n_cancel_lb, 0) AS n_cancels_{lb}d,
               date_diff('day', agg.last_cancel, {c}) AS days_since_last_cancel,
               date_diff('day', agg.last_tx, {c}) AS days_since_last_transaction,
               date_diff('day', agg.first_tx, {c}) AS days_since_first_transaction,
               (agg.msno IS NULL)::INT AS no_transaction_history
        FROM lab l
        LEFT JOIN last USING (msno) LEFT JOIN latest_any USING (msno)
        LEFT JOIN agg USING (msno) LEFT JOIN changes USING (msno)"""
