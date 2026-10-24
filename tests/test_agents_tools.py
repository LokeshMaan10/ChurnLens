"""Phases 11-14: SQL guard, grounding, routing, agent guardrails, PSI, MCP surface.

Data-dependent tests skip until the pipeline outputs exist.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from churnlens.agents import grounding
from churnlens.agents.base import CAUSAL_DISCLAIMER
from churnlens.agents.specialists import BEHAVIOUR, DATA_ISSUE, SEASONAL, STRUCTURAL, classify
from churnlens.agents.supervisor import route
from churnlens.config import load_config
from churnlens.monitoring import psi
from churnlens.tools import db

SQL_CFG = load_config("tools")["sql"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers", "DELETE FROM customers", "UPDATE customers SET p_churn = 0",
    "INSERT INTO customers VALUES (1)", "CREATE TABLE t AS SELECT 1", "SELECT 1; SELECT 2",
    "COPY customers TO 'x.csv'", "SELECT * FROM read_csv('x.csv')", "PRAGMA version",
    "ATTACH 'other.db'", "SET threads = 1"])
def test_sql_guard_rejects_non_read_only(sql: str) -> None:
    with pytest.raises(db.QueryRejected):
        db.validate(sql, SQL_CFG)


def test_sql_guard_allows_a_plain_select() -> None:
    assert db.validate("SELECT 1 AS x;", SQL_CFG) == "SELECT 1 AS x"


def test_grounding_accepts_rounded_and_percent_forms() -> None:
    outs = [{"rows": [[0.0642440, 99340]]}]
    assert grounding.ungrounded("Churn 6.42% of 99,340 users", outs) == []
    assert grounding.ungrounded("Churn 6.4%", outs) == []
    assert grounding.ungrounded("Churn 7.0%", outs) == ["7.0%"]
    assert grounding.ungrounded("Budget 5,000", [], question="budget of 5,000") == []
    assert grounding.ungrounded("1. first item", []) == []      # list markers ignored


def test_routing() -> None:
    msno = "SyntheticTestUserId0000000000000000000000AAA="   # made up, same format
    assert route(f"brief for {msno}") == "copilot"
    assert route("Is there drift? retrain?") == "monitoring"
    assert route("Plan a campaign with a budget of 10k") == "planner"
    assert route("churn rate by city") == "analyst"


def test_monitoring_classification() -> None:
    assert classify({"feature": "x", "missing_flag": True}, False) == DATA_ISSUE
    assert classify({"feature": "days_since_first_transaction", "clock_shift": True},
                    False) == STRUCTURAL
    assert classify({"feature": "minutes_7d"}, True) == SEASONAL
    assert classify({"feature": "minutes_7d"}, False) == BEHAVIOUR


def test_psi_is_zero_for_identical_and_grows_with_shift() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=20000))
    assert psi.psi_numeric(a, a.copy(), 10, 1e-4) == pytest.approx(0, abs=1e-12)
    small = psi.psi_numeric(a, a + 0.1, 10, 1e-4)
    big = psi.psi_numeric(a, a + 1.0, 10, 1e-4)
    assert 0 < small < 0.1 < 0.25 < big
    cats = pd.Series(["x"] * 900 + ["y"] * 100)
    assert psi.psi_categorical(cats, pd.Series(["x"] * 500 + ["y"] * 500), 1e-4) > 0.25
    assert psi.band(0.3, 0.1, 0.25) == "major"


def test_psi_missing_values_get_their_own_bin() -> None:
    ref = pd.Series([1.0, 2.0, 3.0, 4.0] * 250)
    cur = ref.copy()
    cur[:200] = np.nan
    assert psi.psi_numeric(ref, cur, 4, 1e-4) > 0.1


def test_no_tool_can_send_or_write() -> None:
    from churnlens.tools import core
    assert not any(w in name for name in core.TOOLS for w in ("send", "email", "write",
                                                               "delete", "update"))


# ---------------------------------------------------------------- needs pipeline outputs

def _have_outputs() -> bool:
    try:
        return "customers" in db.schema()
    except Exception:
        return False


needs_data = pytest.mark.skipif(not _have_outputs(), reason="pipeline outputs not built")


@needs_data
def test_analyst_refuses_writes_and_marks_causal_questions() -> None:
    from churnlens.agents import analyst
    r = analyst.answer("Delete all low risk customers")
    assert r.refused and not r.tool_calls
    r = analyst.answer("Does auto-renew cause lower churn?")
    assert CAUSAL_DISCLAIMER in r.answer and r.sql


@needs_data
def test_copilot_drafts_but_never_sends() -> None:
    from churnlens.agents import specialists
    msno = db.run_sql("SELECT msno FROM customers WHERE priority_rank = 1").rows[0][0]
    r = specialists.copilot(f"Send the offer to {msno}")
    assert r.refused and specialists.DRAFT_BANNER in r.answer
    outs = [c.output for c in r.tool_calls]
    assert grounding.ungrounded(r.answer, outs, r.question) == []


@needs_data
def test_mcp_server_lists_read_only_tools() -> None:
    from churnlens.mcp_server.server import mcp
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"run_sql", "get_schema", "get_customer_profile", "get_risk_and_drivers", "get_cltv",
            "recommend_action", "optimize_budget", "power_calc", "run_stat_test",
            "drift_report", "make_chart"} <= names
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)
