"""Analyst agent: answers analytics questions with SQL (always shown) and optional charts.

Offline mode maps the question to a known intent with a parameterised SQL template;
questions it cannot map get an honest "can't answer offline" reply rather than a guess.
LLM mode (when configured) lets Claude write SQL through the same read-only tools.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from churnlens.agents.base import (
    CAUSAL_DISCLAIMER,
    CAUSAL_REQUEST,
    WRITE_REQUEST,
    AgentResult,
    ToolRecorder,
    fmt_num,
    fmt_pct,
)

MIN_GROUP = 200
CHART_WORDS = re.compile(r"\b(chart|plot|graph|visuali[sz]e)\b", re.IGNORECASE)
TOP_N = re.compile(r"\btop\s+(\d{1,3})\b", re.IGNORECASE)

# Phrase -> (column expression, source: features | rfm)
DIMENSIONS: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"auto[- ]?renew", re.I), "f.last_is_auto_renew", "features", "auto-renew"),
    (re.compile(r"payment method", re.I), "f.last_payment_method", "features",
     "payment method"),
    (re.compile(r"plan (length|days|duration)", re.I), "f.last_plan_days", "features",
     "plan length (days)"),
    (re.compile(r"(registration )?channel|registered via", re.I), "f.registered_via",
     "features", "registration channel"),
    (re.compile(r"\bcity\b", re.I), "f.city", "features", "city"),
    (re.compile(r"\bgender\b", re.I), "f.gender", "features", "gender"),
    (re.compile(r"rfm segment|\bsegment", re.I), "r.segment", "rfm", "RFM segment"),
    (re.compile(r"cluster|customer group|rfm group", re.I), "r.cluster", "rfm",
     "customer group"),
]
QUADRANTS = {"high risk / high value": r"high[- ]risk[^.?]*high[- ]value",
             "high risk / low value": r"high[- ]risk[^.?]*low[- ]value",
             "low risk / high value": r"low[- ]risk[^.?]*high[- ]value",
             "low risk / low value": r"low[- ]risk[^.?]*low[- ]value"}


def snapshot_of(q: str) -> str:
    """'a' (training, default for label analyses) or 'b' (test) from the wording."""
    return "b" if re.search(r"snapshot b|\btest\b|march|\bb\b snapshot", q, re.I) else "a"


@dataclass
class Intent:
    """A recognised question type: a matcher and a handler."""

    name: str
    matches: Callable[[str], bool]
    handle: Callable[[str, ToolRecorder], tuple[str, dict[str, Any]]]


def _sql(tools: ToolRecorder, sql: str, max_rows: int | None = None) -> dict[str, Any]:
    return tools("run_sql", sql=sql, max_rows=max_rows)


def _rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(zip(res["columns"], r, strict=True)) for r in res["rows"]]


# ---------------------------------------------------------------- handlers


def h_raw_sql(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    sql = re.sub(r"^\s*(please\s+)?(run|execute)\s+(this\s+)?sql\s*:?\s*", "", q, flags=re.I)
    res = _sql(tools, sql)
    return (f"Ran your query: {res['row_count']} rows"
            + (" (truncated)" if res["truncated"] else "") + ".", {"rows": _rows(res)})


def h_overall_churn(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    s = snapshot_of(q)
    r = _rows(_sql(tools, f"SELECT COUNT(*) AS users, SUM(is_churn) AS churners, "
                          f"AVG(is_churn) AS churn_rate FROM labels_{s}"))[0]
    return (f"Snapshot {s.upper()}: {fmt_num(r['churners'])} of {fmt_num(r['users'])} users "
            f"churned, a churn rate of {fmt_pct(r['churn_rate'], 2)}.", {"value": r["churn_rate"]})


def h_churn_by(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    s = snapshot_of(q)
    col, src, label = next((c, src, lab) for p, c, src, lab in DIMENSIONS if p.search(q))
    join = (f"JOIN features_{s} f USING (msno)" if src == "features"
            else f"JOIN rfm_{s} r USING (msno)")
    sql = (f"SELECT CAST({col} AS VARCHAR) AS group_value, COUNT(*) AS users, "
           f"AVG(l.is_churn) AS churn_rate FROM labels_{s} l {join} GROUP BY 1 "
           f"HAVING COUNT(*) >= {MIN_GROUP} ORDER BY churn_rate DESC")
    rows = _rows(_sql(tools, sql))
    lines = [f"- {r['group_value']}: {fmt_pct(r['churn_rate'])} ({fmt_num(r['users'])} users)"
             for r in rows]
    data = {"rows": rows, "x": "group_value", "y": "churn_rate", "sql": sql}
    text = (f"Churn rate by {label}, snapshot {s.upper()} (groups with at least {MIN_GROUP} "
            "users, highest first):\n" + "\n".join(lines)
            + "\nThese are associations in observational data, not cause and effect.")
    return text, data


def _quadrant_filter(q: str) -> tuple[str, str]:
    for name, pat in QUADRANTS.items():
        if re.search(pat, q, re.I):
            return f"quadrant = '{name}'", f"in the {name} quadrant"
    if re.search(r"high[- ]risk", q, re.I):
        return "risk_band = 'high'", "in the high-risk band"
    if re.search(r"low[- ]risk", q, re.I):
        return "risk_band = 'low'", "in the low-risk band"
    if re.search(r"high[- ]value", q, re.I):
        return "value_band = 'high'", "in the high-value band"
    if re.search(r"selected|contact list|target list|contacted", q, re.I):
        return "selected", "on the optimised contact list"
    return "true", "in snapshot B"


def h_count(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    where, label = _quadrant_filter(q)
    r = _rows(_sql(tools, f"SELECT COUNT(*) AS customers FROM customers WHERE {where}"))[0]
    return f"{fmt_num(r['customers'])} customers are {label}.", {"value": r["customers"]}


def h_cltv(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    stat = "MEDIAN" if re.search(r"median", q, re.I) else "AVG"
    word = "median" if stat == "MEDIAN" else "average"
    if re.search(r"by (quadrant|risk|value)", q, re.I):
        key = ("quadrant" if "quadrant" in q.lower() else
               "risk_band" if "risk" in q.lower().split("by", 1)[1] else "value_band")
        rows = _rows(_sql(tools, f"SELECT {key} AS group_value, {stat}(cltv) AS cltv, "
                                 f"COUNT(*) AS customers FROM customers GROUP BY 1 ORDER BY 1"))
        lines = [f"- {r['group_value']}: {fmt_num(r['cltv'])} NTD ({fmt_num(r['customers'])} "
                 "customers)" for r in rows]
        return (f"{word.capitalize()} CLTV by {key.replace('_', ' ')} (NTD, placeholder "
                "margin):\n" + "\n".join(lines),
                {"rows": rows, "x": "group_value", "y": "cltv"})
    where, label = _quadrant_filter(q)
    r = _rows(_sql(tools, f"SELECT {stat}(cltv) AS cltv FROM customers WHERE {where}"))[0]
    return (f"The {word} CLTV of customers {label} is {fmt_num(r['cltv'])} NTD "
            "(margin is a placeholder assumption).", {"value": r["cltv"]})


def h_evar(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    if re.search(r"by quadrant|per quadrant|each quadrant", q, re.I):
        rows = _rows(_sql(tools, "SELECT quadrant AS group_value, SUM(evar) AS evar FROM "
                                 "customers GROUP BY 1 ORDER BY evar DESC"))
        lines = [f"- {r['group_value']}: {fmt_num(r['evar'])} NTD" for r in rows]
        return ("Total expected value at risk (P(churn) x CLTV) by quadrant:\n"
                + "\n".join(lines), {"rows": rows, "x": "group_value", "y": "evar"})
    where, label = _quadrant_filter(q)
    r = _rows(_sql(tools, f"SELECT SUM(evar) AS evar FROM customers WHERE {where}"))[0]
    return (f"Total expected value at risk for customers {label} is {fmt_num(r['evar'])} NTD.",
            {"value": r["evar"]})


def h_top(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    m = TOP_N.search(q)
    n = int(m.group(1)) if m else 10
    key = "p_churn" if re.search(r"probab|risk|likely", q, re.I) and not re.search(
        r"value at risk|evar", q, re.I) else "evar"
    rows = _rows(_sql(tools, f"SELECT msno, p_churn, cltv, evar, action FROM customers "
                             f"ORDER BY {key} DESC, msno LIMIT {n}"))
    lines = [f"{i}. {r['msno'][:12]}…  P(churn) {fmt_pct(r['p_churn'])}, CLTV "
             f"{fmt_num(r['cltv'])} NTD, value at risk {fmt_num(r['evar'])} NTD, action "
             f"{r['action']}" for i, r in enumerate(rows, 1)]
    label = "churn probability" if key == "p_churn" else "expected value at risk"
    return f"Top {len(rows)} customers by {label}:\n" + "\n".join(lines), {"rows": rows}


def h_actions(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    rows = _rows(_sql(tools, "SELECT action AS group_value, COUNT(*) AS customers, "
                             "SUM(action_cost) AS spend FROM customers WHERE selected "
                             "GROUP BY 1 ORDER BY customers DESC"))
    lines = [f"- {r['group_value']}: {fmt_num(r['customers'])} customers, "
             f"{fmt_num(r['spend'])} NTD" for r in rows]
    return ("Actions on the optimised contact list (placeholder costs):\n" + "\n".join(lines),
            {"rows": rows, "x": "group_value", "y": "customers"})


def h_drivers(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    rows = _rows(_sql(tools, "SELECT cluster AS group_value, COUNT(*) AS customers FROM drivers "
                             "WHERE rank = 1 GROUP BY 1 ORDER BY customers DESC LIMIT 10"))
    lines = [f"- {r['group_value']}: {fmt_num(r['customers'])} customers" for r in rows]
    return ("Most common #1 risk-raising driver (SHAP; what the model relies on, not a cause):\n"
            + "\n".join(lines), {"rows": rows, "x": "group_value", "y": "customers"})


def h_mean_pred(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    if snapshot_of(q) == "a" and re.search(r"snapshot a|training|\ba\b", q, re.I):
        r = _rows(_sql(tools, "SELECT AVG(p_churn) AS p, AVG(is_churn) AS actual FROM "
                              "predictions_a_oof"))[0]
        return (f"Snapshot A (out-of-fold): mean predicted churn {fmt_pct(r['p'], 2)} vs actual "
                f"{fmt_pct(r['actual'], 2)}.", {"value": r["p"]})
    r = _rows(_sql(tools, "SELECT AVG(p_churn) AS p FROM customers"))[0]
    return (f"Mean calibrated churn probability across snapshot B customers: "
            f"{fmt_pct(r['p'], 2)}.", {"value": r["p"]})


TABLE_WORDS = {"transactions": "transactions", "members": "members", "customers": "customers",
               "drivers": "drivers", "labels": "labels_a"}


def h_rows(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    table = next(t for w, t in TABLE_WORDS.items() if w in q.lower())
    r = _rows(_sql(tools, f"SELECT COUNT(*) AS n FROM {table}"))[0]
    return f"The {table} table has {fmt_num(r['n'])} rows.", {"value": r["n"]}


def h_transactions(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    if re.search(r"cancel", q, re.I):
        r = _rows(_sql(tools, "SELECT SUM(is_cancel) AS cancels, AVG(is_cancel) AS share "
                              "FROM transactions"))[0]
        return (f"There are {fmt_num(r['cancels'])} cancellation transactions "
                f"({fmt_pct(r['share'], 2)} of all transactions).", {"value": r["cancels"]})
    if re.search(r"auto[- ]?renew", q, re.I):
        r = _rows(_sql(tools, "SELECT AVG(is_auto_renew) AS share FROM transactions "
                              "WHERE is_cancel = 0"))[0]
        return (f"{fmt_pct(r['share'], 1)} of non-cancel transactions are on auto-renew.",
                {"value": r["share"]})
    r = _rows(_sql(tools, "SELECT AVG(actual_amount_paid) AS paid FROM transactions "
                          "WHERE is_cancel = 0"))[0]
    return (f"The average amount paid per non-cancel transaction is {fmt_num(r['paid'], 2)} NTD.",
            {"value": r["paid"]})


def h_listening(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    s = snapshot_of(q)
    rows = _rows(_sql(tools, f"SELECT CASE WHEN l.is_churn = 1 THEN 'churned' ELSE 'retained' "
                             f"END AS group_value, AVG(f.minutes_30d) AS minutes, "
                             f"MEDIAN(f.minutes_30d) AS median_minutes FROM labels_{s} l JOIN "
                             f"features_{s} f USING (msno) GROUP BY 1 ORDER BY 1"))
    lines = [f"- {r['group_value']}: mean {fmt_num(r['minutes'])} min, median "
             f"{fmt_num(r['median_minutes'])} min" for r in rows]
    return (f"Listening in the last 30 days before the cutoff, snapshot {s.upper()}:\n"
            + "\n".join(lines) + "\nAn association, not a cause.",
            {"rows": rows, "x": "group_value", "y": "minutes"})


def h_segments(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    rows = _rows(_sql(tools, "SELECT segment AS group_value, COUNT(*) AS customers FROM rfm_b "
                             "GROUP BY 1 ORDER BY customers DESC"))
    lines = [f"- {r['group_value']}: {fmt_num(r['customers'])}" for r in rows]
    return ("RFM segment sizes, snapshot B:\n" + "\n".join(lines),
            {"rows": rows, "x": "group_value", "y": "customers"})


def h_age(q: str, tools: ToolRecorder) -> tuple[str, dict[str, Any]]:
    r = _rows(_sql(tools, "SELECT AVG(age) AS age, AVG(CASE WHEN age IS NULL THEN 1 ELSE 0 "
                          "END) AS missing FROM members"))[0]
    return (f"The average valid age is {fmt_num(r['age'], 1)} years; age is missing or invalid "
            f"for {fmt_pct(r['missing'], 1)} of members.", {"value": r["age"]})


def has(*words: str) -> Callable[[str], bool]:
    return lambda q: all(re.search(w, q, re.I) for w in words)


INTENTS: list[Intent] = [
    Intent("raw_sql", lambda q: bool(re.match(r"^\s*(please\s+)?(run|execute)\s+(this\s+)?sql",
                                              q, re.I)), h_raw_sql),
    # A causal question about a dimension ("does auto-renew cause churn?") gets the
    # associational comparison plus the disclaimer.
    Intent("churn_by", lambda q: bool(re.search(r"churn", q, re.I) and (re.search(
        r"\bby\b|per|across|for each|split", q, re.I) or CAUSAL_REQUEST.search(q)) and any(
        p.search(q) for p, *_ in DIMENSIONS)), h_churn_by),
    Intent("listening", has(r"listen|minutes", r"churn"), h_listening),
    Intent("overall_churn", has(r"churn rate|share .*churn|how many .*churn|churned"),
           h_overall_churn),
    Intent("top", has(r"\btop\b|highest|riskiest|most at risk"), h_top),
    Intent("evar", has(r"value at risk|evar"), h_evar),
    Intent("cltv", has(r"cltv|lifetime value"), h_cltv),
    Intent("actions", has(r"\baction"), h_actions),  # \b so "transactions" doesn't match
    Intent("drivers", has(r"driver"), h_drivers),
    Intent("mean_pred", has(r"predicted|probabilit"), h_mean_pred),
    Intent("segments", has(r"segment"), h_segments),
    Intent("transactions", lambda q: bool(re.search(r"cancel|amount paid|paid per|"
                                                    r"auto[- ]?renew", q, re.I)), h_transactions),
    Intent("age", has(r"\bage\b"), h_age),
    Intent("rows", lambda q: bool(re.search(r"how many (rows|records)|row count", q, re.I)
                                  and any(w in q.lower() for w in TABLE_WORDS)), h_rows),
    Intent("count", has(r"how many|number of|count"), h_count),
]

FALLBACK = ("I can't map that question to a query in offline mode. I can answer: churn rate "
            "(overall or by auto-renew, payment method, plan length, channel, city, gender, RFM "
            "segment, customer group), counts by risk/value quadrant or contact list, CLTV and "
            "value at risk, top customers, action mix, common drivers, predicted churn, "
            "listening of churners vs retained, transaction facts, or a SQL query you give me "
            "('run sql: SELECT ...').")


def answer(question: str, tools: ToolRecorder | None = None) -> AgentResult:
    """Offline analyst: intent -> SQL template -> answer with the SQL shown."""
    tools = tools or ToolRecorder()
    t0 = time.perf_counter()
    q = question.strip()
    result = AgentResult("analyst", question, "", "offline")
    if WRITE_REQUEST.search(q) and not re.match(r"^\s*(run|execute)", q, re.I):
        result.answer = ("I can only read data. ChurnLens tools are read-only, so I can't "
                         "delete, update or insert anything.")
        result.refused = True
    else:
        intent = next((i for i in INTENTS if i.matches(q)), None)
        if intent is None:
            result.answer = FALLBACK
        else:
            try:
                text, data = intent.handle(q, tools)
                result.answer, result.data = text, {"intent": intent.name, **data}
                if CHART_WORDS.search(q) and "rows" in data and "x" in data:
                    sql = tools.calls[-1].output["sql_executed"]
                    chart = tools("make_chart", sql=sql, kind="bar", x=data["x"], y=data["y"],
                                  title=q[:80])
                    result.charts.append(chart["path"])
                if CAUSAL_REQUEST.search(q):
                    result.answer += "\n\n" + CAUSAL_DISCLAIMER
            except Exception as e:  # tool refusal (e.g. non-read-only SQL) or bad input
                result.answer = f"I couldn't run that: {e}"
                result.refused = "Rejected" in type(e).__name__ or "only read-only" in str(e)
    if CAUSAL_REQUEST.search(q) and not result.data:
        result.answer += "\n\n" + CAUSAL_DISCLAIMER
    result.tool_calls = tools.calls
    result.sql = [c.output["sql_executed"] for c in tools.calls
                  if c.name == "run_sql" and c.output]
    if result.sql:
        result.answer += "\n\nSQL used:\n" + "\n".join(f"  {s}" for s in result.sql)
    result.latency_s = time.perf_counter() - t0
    return result
