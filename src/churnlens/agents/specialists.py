"""Retention copilot, campaign planner and monitoring agent (offline implementations).

Each composes read-only tools and renders their outputs; every number shown comes from a
tool result. The copilot drafts outreach text but has no way to send it.
"""

from __future__ import annotations

import re
import time
from typing import Any

from churnlens.agents.base import (
    CAUSAL_DISCLAIMER,
    MSNO_IN_TEXT,
    SEND_REQUEST,
    AgentResult,
    ToolRecorder,
    fmt_num,
    fmt_pct,
)
from churnlens.config import load_config, resolve_path
from churnlens.tools.core import ToolError

DRAFT_BANNER = ("DRAFT ONLY — not sent. ChurnLens has no sending capability; a person must "
                "review and send.")
MONEY = re.compile(r"(?:budget|spend)\D{0,20}([\d,]+(?:\.\d+)?)\s*(k|thousand)?", re.I)
CAPACITY = re.compile(r"([\d,]+)\s*(?:contacts|customers|users|people|calls|messages)", re.I)
CAP_WORDS = re.compile(r"(?:capacity|reach|contact)\D{0,15}([\d,]+)", re.I)

OFFER_LINES = {
    "auto_renew_incentive": "If you switch on auto-renew this week, we'll add a bonus month "
                            "to your plan.",
    "auto_renew_reminder": "Switching on auto-renew keeps your music playing without "
                           "interruptions.",
    "payment_support": "If anything went wrong with your payment method, we can help you "
                       "update it in a minute.",
    "win_back_offer": "We'd love to have you back — here's a discount on your next renewal.",
    "cancellation_survey": "Could you tell us what didn't work for you? It takes a minute.",
    "trial_conversion_offer": "Your first paid month is on us at a special price.",
    "plan_upgrade_offer": "A longer plan could save you money — here's an offer for you.",
    "content_reengagement": "We picked new releases and playlists we think you'll like.",
    "general_retention_message": "We miss you — here's what's new on KKBox.",
    "monitor": "",
}


# ---------------------------------------------------------------- retention copilot


def copilot(question: str, tools: ToolRecorder | None = None) -> AgentResult:
    """Customer brief + recommended action + a draft message (never sent)."""
    tools = tools or ToolRecorder()
    t0 = time.perf_counter()
    res = AgentResult("copilot", question, "", "offline")
    m = MSNO_IN_TEXT.search(question)
    if not m:
        res.answer = "Give me a customer id (msno) and I'll write a brief for them."
        res.latency_s = time.perf_counter() - t0
        return res
    msno = m.group(0)
    try:
        prof = tools("get_customer_profile", msno=msno)
        risk = tools("get_risk_and_drivers", msno=msno)
        val = tools("get_cltv", msno=msno)
        act = tools("recommend_action", msno=msno)
    except ToolError as e:
        res.answer = f"I couldn't build a brief: {e}"
        res.tool_calls, res.latency_s = tools.calls, time.perf_counter() - t0
        return res
    p = prof["profile"]
    auto = {1: "on auto-renew", 0: "not on auto-renew"}.get(p["last_is_auto_renew"], "no plan")
    drivers = "\n".join(f"  {d['rank']}. {d['reason']}" for d in risk["drivers"][:5]) \
        or "  (no risk-raising drivers above the reporting floor)"
    sent = SEND_REQUEST.search(question)
    offer = OFFER_LINES.get(act["action"], "")
    draft = "" if act["action"] == "monitor" else (
        f"\n\nDraft outreach ({act['action']}):\n  Hi there, thanks for listening with KKBox. "
        f"{offer}\n  {DRAFT_BANNER}")
    res.answer = (
        f"Customer brief — {msno}\n"
        f"- Churn risk: {fmt_pct(risk['p_churn'])} (calibrated), {risk['risk_band']} risk, "
        f"quadrant: {risk['quadrant']}\n"
        f"- Plan: {fmt_num(p['last_plan_days'])}-day plan at {fmt_num(p['last_amount_paid'])} NTD, "
        f"{auto}, payment method {p['last_payment_method']}\n"
        f"- Listening: {fmt_num(p['minutes_30d'])} minutes over {fmt_num(p['active_days_30d'])} "
        f"active days in the last 30 days\n"
        f"- {fmt_num(val['horizon_months'])}-month CLTV if retained: "
        f"{fmt_num(val['cltv_used'])} NTD; expected value at risk "
        f"{fmt_num(val['expected_value_at_risk'])} NTD (placeholder margin)\n"
        f"- Top drivers (what the model relies on — associations, not cause and effect):\n"
        f"{drivers}\n"
        f"- Recommended action: {act['action']} — {act['action_description']} (rule: "
        f"{act['rule']}; expected benefit {fmt_num(act['expected_benefit'])} NTD under "
        f"placeholder costs; on the current contact list: "
        f"{'yes' if act['selected'] else 'no'})" + draft)
    if sent:
        res.answer = ("I can't send messages — ChurnLens only drafts them for a person to "
                      "review.\n\n") + res.answer
        res.refused = True
    res.data = {"msno": msno, "p_churn": risk["p_churn"], "action": act["action"]}
    res.tool_calls, res.latency_s = tools.calls, time.perf_counter() - t0
    return res


# ---------------------------------------------------------------- campaign planner


def _amount(text: str) -> float | None:
    m = MONEY.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * 1000 if m.group(2) else v


def _capacity(text: str) -> int | None:
    for pat in (CAPACITY, CAP_WORDS):
        m = pat.search(text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def planner(question: str, tools: ToolRecorder | None = None) -> AgentResult:
    """Target list, action mix and an A/B test design with a randomized holdout."""
    tools = tools or ToolRecorder()
    t0 = time.perf_counter()
    cfg = load_config("agents")["planner"]
    budget_cfg = load_config("prioritize")["budget"]
    res = AgentResult("planner", question, "", "offline")
    budget = _amount(question) or float(budget_cfg["total_ntd"])
    cap = _capacity(question) or int(budget_cfg["outreach_capacity"])
    opt = tools("optimize_budget", budget_ntd=budget, outreach_capacity=cap,
                top_n=int(cfg["top_n"]))
    n = opt["contacted"]
    # Expected churn of the selected group: mean calibrated P over the re-optimised list.
    base = opt["mean_p_churn_selected"] or 0.0
    ab = tools("design_ab_test", n_selected=n, holdout_share=float(cfg["holdout_share"]),
               baseline_rate=round(base, 4),
               relative_reduction=float(cfg["assumed_relative_reduction"]),
               alpha=float(cfg["alpha"]), power=float(cfg["power"])) \
        if n > 0 and 0 < base < 1 else None
    mix = "\n".join(f"  - {a}: {fmt_num(v['users'])} customers, {fmt_num(v['spend'])} NTD"
                    for a, v in opt["action_mix"].items())
    top = "\n".join(f"  {i}. {t['msno'][:12]}…  P(churn) {fmt_pct(t['p_churn'])}, action "
                    f"{t['action']}, expected benefit {fmt_num(t['expected_benefit'])} NTD"
                    for i, t in enumerate(opt["top"][:10], 1))
    if ab:
        verdict = "sufficient" if ab["sufficient"] else \
            "NOT sufficient: enlarge the list or the holdout"
        design = (f"- Randomize the {fmt_num(ab['n_selected'])} selected customers: "
                  f"{fmt_num(ab['n_contacted'])} contacted, {fmt_num(ab['n_holdout'])} held out "
                  f"({fmt_num(ab['holdout_percent'])}%, not contacted).\n"
                  f"- Baseline churn (mean calibrated P of the selected customers): "
                  f"{fmt_pct(ab['baseline_rate'])}; powered for a "
                  f"{fmt_num(ab['relative_reduction_percent'])}% relative reduction (assumption) "
                  f"at alpha {ab['alpha']} and power {fmt_pct(ab['power'], 0)}: needs "
                  f"{fmt_num(ab['required_holdout'])} held out + "
                  f"{fmt_num(ab['required_contacted'])} contacted — {verdict} (achieved power "
                  f"{fmt_pct(ab['achieved_power'], 0)}).\n"
                  "- Primary metric: churn at the next expiry; check the sample-ratio split "
                  "before reading results; analyse with the two-proportion test.")
    else:
        design = "- Could not size the test (no selected customers or no baseline churn)."
    res.answer = (f"Campaign plan — budget {fmt_num(budget)} NTD, capacity {fmt_num(cap)} "
                  "contacts\n"
                  f"- Contact {fmt_num(n)} customers, spend {fmt_num(opt['spend'])} NTD, expected "
                  f"net benefit {fmt_num(opt['expected_net_benefit'])} NTD (upper bound "
                  f"{fmt_num(opt['upper_bound'])} NTD; placeholder costs and save rates).\n"
                  f"- Action mix:\n{mix}\n- Top of the list:\n{top}\n\n"
                  f"A/B test design:\n{design}\n\n"
                  "Holding out at-risk customers means some likely churners are not contacted; "
                  "that is the price of measuring the real effect. " + CAUSAL_DISCLAIMER)
    res.data = {"contacted": n, "holdout": ab["n_holdout"] if ab else None,
                "expected_net_benefit": opt["expected_net_benefit"]}
    res.tool_calls, res.latency_s = tools.calls, time.perf_counter() - t0
    return res


# ---------------------------------------------------------------- monitoring agent

DATA_ISSUE = "data issue"
STRUCTURAL = "structural (clock) shift"
SEASONAL = "likely calendar / seasonal"
BEHAVIOUR = "behaviour shift"
SHORT_WINDOW = re.compile(r"_(7|14)d|_ratio_7_|days_since_last_listen|r_score")


def classify(feature: dict[str, Any], events_overlap: bool) -> str:
    """Separate data problems from real shifts: missing-share jumps are data issues;
    time-since features whose median moved by the gap between snapshots are structural
    (the calendar moved, often because history is left-censored); short-window listening
    features moving while a calendar event overlaps the reference window are likely
    seasonal; everything else is treated as a behaviour shift."""
    if feature.get("missing_flag"):
        return DATA_ISSUE
    if feature.get("clock_shift"):
        return STRUCTURAL
    if events_overlap and SHORT_WINDOW.search(feature["feature"]):
        return SEASONAL
    return BEHAVIOUR


def monitoring(question: str = "Investigate drift", tools: ToolRecorder | None = None,
               write_note: bool = True) -> AgentResult:
    """Investigate drift alerts and write an incident note (scheduled via `make monitor`)."""
    tools = tools or ToolRecorder()
    t0 = time.perf_counter()
    mon = load_config("monitoring")
    res = AgentResult("monitoring", question, "", "offline")
    try:
        rep = tools("drift_report", top_n=1000)
    except ToolError as e:
        res.answer = f"No drift report available: {e}"
        res.latency_s = time.perf_counter() - t0
        return res
    overlap = bool(rep.get("calendar_events_in_reference_window"))
    flagged = [f for f in rep["features"] if f["band"] != "stable" or f.get("missing_flag")]
    groups: dict[str, list[dict[str, Any]]] = {DATA_ISSUE: [], STRUCTURAL: [], SEASONAL: [],
                                               BEHAVIOUR: []}
    for f in flagged:
        groups[classify(f, overlap)].append(f)
    pred = rep["prediction"]
    verdict = rep["retrain"]
    lines = [f"# Incident note — drift {rep['reference']} → {rep['current']}",
             f"_Monitoring agent (offline mode), from the drift report generated "
             f"{rep['generated']}._", "",
             f"**Prediction drift:** PSI {pred['psi']:.3f} ({pred['band']}). Mean score "
             f"{fmt_pct(pred['reference_mean'], 2)} → {fmt_pct(pred['current_mean'], 2)}.",
             f"**Features:** {rep['n_features']} monitored; {len(flagged)} flagged "
             f"({rep['n_major']} major, {rep['n_moderate']} moderate, "
             f"{rep['n_missing_flags']} missing-share flags).", ""]
    for title, items in groups.items():
        lines.append(f"## {title.capitalize()} ({len(items)})")
        if not items:
            lines += ["- none", ""]
            continue
        for f in items[:10]:
            med = (f", median {fmt_num(f['median_ref'])} → {fmt_num(f['median_cur'])}"
                   if f.get("median_ref") is not None else "")
            lines.append(f"- `{f['feature']}`: PSI {f['psi']:.3f} ({f['band']}), missing "
                         f"{fmt_pct(f['missing_ref'], 1)} → {fmt_pct(f['missing_cur'], 1)}{med}")
        if title == STRUCTURAL:
            lines.append(f"  For customers present in both snapshots the value moved by "
                         f"{fmt_num(items[0]['per_user_median_change'])} days (median), the "
                         f"{rep['cutoff_gap_days']}-day gap between cutoffs: it tracks the "
                         "calendar, not behaviour. If the model relies on it, cap it or "
                         "measure it from a fixed point, then retrain.")
        lines.append("")
    if overlap:
        ev = ", ".join(rep["calendar_events_in_reference_window"])
        lines += [f"Calendar context: the reference window overlaps {ev}; short-window "
                  "listening features are expected to differ for that reason (a timing "
                  "coincidence, not a tested cause).", ""]
    reasons = "; ".join(verdict["reasons"]) or "no trigger met"
    lines += ["## Retraining decision",
              f"**{'RETRAIN' if verdict['retrain'] else 'No retrain needed'}** — {reasons}.", "",
              "## Next steps",
              "- Data issues: check the pipeline/source for the flagged columns before trusting "
              "scores that depend on them.",
              "- Behaviour shifts: confirm with fresh labels (calibration gap, PR-AUC) per the "
              "retraining policy in `configs/monitoring.yaml`."]
    note = "\n".join(lines)
    if write_note:
        path = resolve_path(mon["outputs"]["incident_note"], load_config("data"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note + "\n", encoding="utf-8")
    res.answer = note
    res.data = {"groups": {k: [f["feature"] for f in v] for k, v in groups.items()},
                "retrain": verdict["retrain"]}
    res.tool_calls, res.latency_s = tools.calls, time.perf_counter() - t0
    return res
