"""Power BI exports: clean, typed tables for a star-schema report.

Run with `python -m churnlens.scoring.exports` (or `make exports`). Writes CSV (and
Parquet) to data/exports/{mode}/ — under data/, so never committed. See
docs/powerbi_guide.md for the model and measures.
"""

from __future__ import annotations

import json

import pandas as pd

from churnlens.config import load_config, resolve_path
from churnlens.tools import db

TABLES = {
    # fact: one row per snapshot B customer
    "fact_customer_scores": """
        SELECT c.msno AS customer_id, c.p_churn, c.cltv, c.evar AS value_at_risk,
               c.risk_band, c.value_band, c.quadrant, c.action, c.action_cost,
               c.expected_benefit, c.selected AS on_contact_list, c.priority_rank,
               c.driver_cluster AS top_driver, c.driver_reason AS top_driver_reason,
               l.is_churn AS actual_churn
        FROM customers c JOIN labels_b l ON c.msno = l.msno""",
    # fact: top drivers (long)
    "fact_customer_drivers": """
        SELECT msno AS customer_id, rank, cluster AS driver, feature, "group" AS driver_group,
               value, contribution, reason FROM drivers""",
    # dimension: customer attributes (no demographics beyond what the app shows)
    "dim_customer": """
        SELECT f.msno AS customer_id, f.last_plan_days, f.last_amount_paid,
               f.last_is_auto_renew, f.last_payment_method, f.registered_via, f.tenure_days,
               f.minutes_30d, f.active_days_30d, r.segment AS rfm_segment,
               r.cluster AS customer_group, r.r_score, r.f_score, r.m_score
        FROM features_b f JOIN rfm_b r ON f.msno = r.msno""",
}


def main() -> int:
    """Export the star-schema tables plus a small KPI table."""
    data_cfg = load_config("data")
    out = resolve_path("data/exports/{mode}", data_cfg)
    out.mkdir(parents=True, exist_ok=True)
    con = db.connection().cursor()
    rows = {}
    for name, sql in TABLES.items():
        df = con.execute(sql).df()
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.8g")
        df.to_parquet(out / f"{name}.parquet", index=False)
        rows[name] = len(df)
    summary = json.loads(resolve_path("reports/model/summary.json", data_cfg)
                         .read_text(encoding="utf-8"))
    test = summary["models"][summary["selected"]]["test"]
    pct = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    kpis = pd.DataFrame([
        {"kpi": "model_pr_auc_B", "value": test["pr_auc"]},
        {"kpi": "model_roc_auc_B", "value": test["roc_auc"]},
        {"kpi": "churn_rate_B", "value": test["base_rate"]},
        {"kpi": "contact_threshold", "value": summary["threshold"]},
        {"kpi": "data_mode_sample_pct", "value": pct},
    ])
    kpis.to_csv(out / "kpis.csv", index=False, float_format="%.8g")
    print(f"exports -> {out}: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
