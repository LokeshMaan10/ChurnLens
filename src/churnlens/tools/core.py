"""Pure tool functions exposed to agents (via MCP) and to the app.

Every tool returns a JSON-serialisable dict. Tools only read; none can send messages or
modify data. Numbers in agent answers must come from these outputs (Phase 13 checks it).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from churnlens.analysis.plotting import SERIES, apply_style  # noqa: E402
from churnlens.config import load_config, resolve_path  # noqa: E402
from churnlens.experiments import stats as xs  # noqa: E402
from churnlens.prioritize import engine  # noqa: E402
from churnlens.tools import db  # noqa: E402

MSNO_PATTERN = re.compile(r"^[A-Za-z0-9+/=]{20,60}$")
PROFILE_COLUMNS = ["last_plan_days", "last_list_price", "last_amount_paid", "last_is_auto_renew",
                   "last_payment_method", "days_to_expiry", "n_transactions_365d",
                   "total_paid_365d", "n_cancels_365d", "tenure_days", "registered_via",
                   "minutes_30d", "active_days_30d", "days_since_last_listen",
                   "minutes_ratio_30_60", "skip_rate_30d", "rfm_segment", "rfm_group"]


class ToolError(ValueError):
    """Bad tool input (unknown customer, invalid parameter)."""


def _one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    cur = db.connection().cursor()
    rel = cur.execute(sql, params)
    cols = [d[0] for d in rel.description]
    row = rel.fetchone()
    return None if row is None else {c: db._jsonable(v) for c, v in zip(cols, row, strict=True)}


def _check_msno(msno: str) -> str:
    msno = msno.strip()
    if not MSNO_PATTERN.match(msno):
        raise ToolError("msno must be a KKBox user id (base64-like, 20-60 characters)")
    return msno


# ---------------------------------------------------------------- data tools


def run_sql(sql: str, max_rows: int | None = None) -> dict[str, Any]:
    """Read-only SQL over the curated views. Returns the SQL that ran and the rows."""
    return db.run_sql(sql, max_rows=max_rows).as_dict()


def get_schema(table: str | None = None) -> dict[str, Any]:
    """Curated views: description and columns (all views, or one)."""
    return db.schema(table)


def get_customer_profile(msno: str) -> dict[str, Any]:
    """Plan, payment, tenure and listening facts for one snapshot B customer."""
    msno = _check_msno(msno)
    cols = ", ".join(PROFILE_COLUMNS)
    row = _one(f"SELECT {cols} FROM features_b WHERE msno = ?", (msno,))
    if row is None:
        raise ToolError(f"customer {msno} not found in snapshot B")
    rfm = _one("SELECT recency_days, frequency_days, monetary, r_score, f_score, m_score "
               "FROM rfm_b WHERE msno = ?", (msno,))
    return {"msno": msno, "snapshot": "B", "profile": row, "rfm": rfm,
            "source": "features_b, rfm_b"}


def get_risk_and_drivers(msno: str) -> dict[str, Any]:
    """Calibrated churn probability, risk band and top risk-raising drivers."""
    msno = _check_msno(msno)
    risk = _one("SELECT p_churn, risk_band, quadrant FROM customers WHERE msno = ?", (msno,))
    if risk is None:
        raise ToolError(f"customer {msno} not found in snapshot B")
    rows = db.connection().cursor().execute(
        "SELECT rank, cluster, feature, value, contribution, reason FROM drivers "
        "WHERE msno = ? ORDER BY rank", (msno,)).fetchall()
    drivers = [{"rank": r[0], "cluster": r[1], "feature": r[2], "value": r[3],
                "contribution_log_odds": round(float(r[4]), 4), "reason": r[5]} for r in rows]
    return {"msno": msno, **risk, "drivers": drivers,
            "note": "Drivers describe what the model relies on (SHAP); they are associations, "
                    "not causes.", "source": "customers, drivers"}


def get_cltv(msno: str) -> dict[str, Any]:
    """CLTV (12 months, if the customer renews now) by both methods, and the one in use."""
    msno = _check_msno(msno)
    row = _one("SELECT monthly_price, margin_month, formula_segment, cltv_formula, "
               "cltv_survival FROM cltv WHERE msno = ?", (msno,))
    if row is None:
        raise ToolError(f"customer {msno} not found in snapshot B")
    used = _one("SELECT cltv, evar FROM customers WHERE msno = ?", (msno,))
    return {"msno": msno, **row, "horizon_months": load_config("value")["horizon_months"],
            "cltv_used": used["cltv"], "expected_value_at_risk":
            used["evar"], "units": "NTD; margin and discount rate are placeholder assumptions",
            "source": "cltv, customers"}


def recommend_action(msno: str) -> dict[str, Any]:
    """Rule-engine action for a customer, its cost, expected benefit and list status."""
    msno = _check_msno(msno)
    row = _one("SELECT rule, action, action_cost, expected_benefit, selected, priority_rank, "
               "driver_reason FROM customers WHERE msno = ?", (msno,))
    if row is None:
        raise ToolError(f"customer {msno} not found in snapshot B")
    act = load_config("prioritize")["actions"][row["action"]]
    return {"msno": msno, **row, "action_description": act["description"],
            "save_rate_assumption": act["save_rate"],
            "note": "Costs and save rates are placeholder assumptions.",
            "source": "customers, configs/prioritize.yaml"}


def optimize_budget(budget_ntd: float, outreach_capacity: int, top_n: int = 20
                    ) -> dict[str, Any]:
    """Re-run the Phase 9 optimizer for a new budget / capacity over snapshot B."""
    if budget_ntd < 0 or outreach_capacity < 0:
        raise ToolError("budget and capacity must be non-negative")
    cfg = load_config("prioritize")["budget"]
    df = db.connection().cursor().execute(
        "SELECT msno, p_churn, cltv, evar, action, action_cost, expected_benefit, "
        "driver_reason FROM customers ORDER BY msno").df()
    cost, ben = df["action_cost"].to_numpy(float), df["expected_benefit"].to_numpy(float)
    lambdas = np.concatenate([[0.0], np.logspace(np.log10(cfg["lambda_min"]),
                                                 np.log10(cfg["lambda_max"]),
                                                 int(cfg["lambda_grid_points"]))])
    chosen = engine.greedy_select(cost, ben, float(budget_ntd), int(outreach_capacity), lambdas)
    sel = df[chosen].sort_values("expected_benefit", ascending=False)
    mix = sel.groupby("action").agg(users=("msno", "size"), spend=("action_cost", "sum"))
    return {"budget_ntd": budget_ntd, "outreach_capacity": outreach_capacity,
            "contacted": int(chosen.sum()), "spend": round(float(cost[chosen].sum()), 2),
            "mean_p_churn_selected": round(float(df.loc[chosen, "p_churn"].mean()), 6)
            if chosen.any() else None,
            "expected_net_benefit": round(float(ben[chosen].sum()), 2),
            "upper_bound": round(engine.lp_upper_bound(cost, ben, float(budget_ntd),
                                                       int(outreach_capacity)), 2),
            "action_mix": {a: {"users": int(r.users), "spend": float(r.spend)}
                           for a, r in mix.iterrows()},
            "top": [{"msno": r.msno, "p_churn": round(r.p_churn, 4), "cltv": round(r.cltv, 2),
                     "action": r.action, "expected_benefit": round(r.expected_benefit, 2),
                     "reason": r.driver_reason} for r in sel.head(top_n).itertuples()],
            "note": "Expected values use placeholder costs and save rates.",
            "source": "customers + Phase 9 optimizer"}


# ---------------------------------------------------------------- experiment tools


def power_calc(baseline_rate: float, relative_reduction: float, alpha: float = 0.05,
               power: float = 0.8, treatment_share: float = 0.5) -> dict[str, Any]:
    """Users needed per arm to detect a relative churn reduction (two-sided z-test)."""
    if not (0 < baseline_rate < 1 and 0 < relative_reduction < 1 and 0 < treatment_share < 1):
        raise ToolError("rates and shares must be strictly between 0 and 1")
    k = treatment_share / (1 - treatment_share)
    p_t = baseline_rate * (1 - relative_reduction)
    n_c, n_t = xs.sample_size(baseline_rate, p_t, alpha, power, k)
    return {"baseline_rate": baseline_rate, "treated_rate": round(p_t, 6),
            "relative_reduction": relative_reduction, "alpha": alpha, "power": power,
            "treatment_share": treatment_share, "n_control": n_c, "n_treatment": n_t,
            "n_total": n_c + n_t, "source": "two-proportion sample-size formula"}


def design_ab_test(n_selected: int, holdout_share: float, baseline_rate: float,
                   relative_reduction: float, alpha: float = 0.05,
                   power: float = 0.8) -> dict[str, Any]:
    """Split a selected list into contacted / randomized holdout and check whether the split
    can detect the assumed relative churn reduction."""
    if n_selected <= 0 or not 0 < holdout_share < 1:
        raise ToolError("n_selected must be positive and holdout_share in (0, 1)")
    n_hold = int(round(n_selected * holdout_share))
    n_treat = n_selected - n_hold
    req = power_calc(baseline_rate, relative_reduction, alpha, power, 1 - holdout_share)
    achieved = xs.achieved_power(baseline_rate, req["treated_rate"], max(n_hold, 1),
                                 max(n_treat, 1), alpha)
    return {"n_selected": n_selected, "holdout_share": holdout_share,
            "holdout_percent": round(holdout_share * 100, 4), "n_holdout": n_hold,
            "n_contacted": n_treat, "baseline_rate": baseline_rate,
            "relative_reduction": relative_reduction,
            "relative_reduction_percent": round(relative_reduction * 100, 4),
            "required_holdout": req["n_control"], "required_contacted": req["n_treatment"],
            "achieved_power": round(achieved, 4),
            "sufficient": bool(n_hold >= req["n_control"] and n_treat >= req["n_treatment"]),
            "alpha": alpha, "power": power, "source": "two-proportion power formulas"}


def run_stat_test(churned_control: int, n_control: int, churned_treatment: int,
                  n_treatment: int, alpha: float = 0.05,
                  planned_treatment_share: float | None = None) -> dict[str, Any]:
    """Two-proportion z-test with CI, plus an optional sample-ratio-mismatch check."""
    if min(n_control, n_treatment) <= 0 or churned_control > n_control \
            or churned_treatment > n_treatment:
        raise ToolError("counts must satisfy 0 <= churned <= n and n > 0")
    out: dict[str, Any] = xs.two_proportion_test(churned_control, n_control, churned_treatment,
                                                 n_treatment, alpha).as_dict()
    if planned_treatment_share is not None:
        out["srm"] = xs.srm_check(n_control, n_treatment, planned_treatment_share,
                                  load_config("experiments")["srm_alpha"])
    out["source"] = "two-proportion z-test"
    return out


# ---------------------------------------------------------------- monitoring + charts


def drift_report(top_n: int = 15) -> dict[str, Any]:
    """Latest Phase 14 drift results (PSI per feature/prediction, retraining verdict)."""
    path = resolve_path(load_config("monitoring")["outputs"]["summary_json"],
                        load_config("data"))
    if not Path(path).exists():
        raise ToolError("no drift report yet: run `python -m churnlens.monitoring.run`")
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    rep["features"] = rep["features"][:top_n]
    rep["source"] = str(Path(path).name)
    return rep


def make_chart(sql: str, kind: str, x: str, y: str, title: str) -> dict[str, Any]:
    """Run a read-only query and save a bar or line chart of column y by x as a PNG."""
    if kind not in ("bar", "line"):
        raise ToolError("kind must be 'bar' or 'line'")
    cfg = load_config("tools")["charts"]
    res = db.run_sql(sql, max_rows=int(cfg["max_points"]))
    if x not in res.columns or y not in res.columns:
        raise ToolError(f"columns {x!r}/{y!r} not in query result {res.columns}")
    xs_ = [str(r[res.columns.index(x)]) for r in res.rows]
    ys = [float(r[res.columns.index(y)]) for r in res.rows]
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "bar":
        ax.bar(xs_, ys, color=SERIES[0])
    else:
        ax.plot(xs_, ys, color=SERIES[0], marker="o")
    ax.set_title(title, loc="left")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_dir = resolve_path(cfg["dir"], load_config("data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"chart_{stamp}.png"
    fig.savefig(path, dpi=int(cfg["dpi"]))
    plt.close(fig)
    return {"path": str(path), "sql_executed": res.sql_executed, "points": len(xs_),
            "data": {"x": xs_, "y": ys}, "source": "run_sql + matplotlib"}


TOOLS = {f.__name__: f for f in (run_sql, get_schema, get_customer_profile,
                                 get_risk_and_drivers, get_cltv, recommend_action,
                                 optimize_budget, power_calc, design_ab_test, run_stat_test,
                                 drift_report, make_chart)}
