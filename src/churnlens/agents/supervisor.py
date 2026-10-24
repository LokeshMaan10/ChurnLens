"""Supervisor: routes each request to one agent, in offline or LLM mode.

Run from the command line:  python -m churnlens.agents.supervisor "your question"
"""

from __future__ import annotations

import re
import sys

from churnlens.agents import analyst, prompts, specialists
from churnlens.agents.base import MSNO_IN_TEXT, AgentResult, resolve_mode, write_trace
from churnlens.config import load_config

PLANNER_WORDS = re.compile(r"\b(budget|campaign|target list|a/b|ab test|holdout|"
                           r"who should we contact|outreach plan)\b", re.I)
MONITOR_WORDS = re.compile(r"\b(drift|psi|monitor|incident|data quality|retrain)", re.I)
TOOLSETS = {
    "analyst": ["get_schema", "run_sql", "make_chart"],
    "copilot": ["get_customer_profile", "get_risk_and_drivers", "get_cltv", "recommend_action"],
    "planner": ["optimize_budget", "power_calc", "run_sql"],
    "monitoring": ["drift_report"],
}
SYSTEMS = {"analyst": prompts.ANALYST, "copilot": prompts.COPILOT, "planner": prompts.PLANNER,
           "monitoring": prompts.MONITORING}


def route(question: str) -> str:
    """Rule-based routing (offline mode, and the fallback for LLM mode)."""
    if MSNO_IN_TEXT.search(question):
        return "copilot"
    if MONITOR_WORDS.search(question):
        return "monitoring"
    if PLANNER_WORDS.search(question):
        return "planner"
    return "analyst"


def ask(question: str, trace: bool = True) -> AgentResult:
    """Route and answer one request; every run is appended to the JSON trace log."""
    mode = resolve_mode(load_config("agents"))
    agent = route(question)
    if mode == "llm":
        from churnlens.agents.llm import run_claude
        result = run_claude(agent, SYSTEMS[agent], TOOLSETS[agent], question)
    else:
        result = {"analyst": analyst.answer, "copilot": specialists.copilot,
                  "planner": specialists.planner,
                  "monitoring": specialists.monitoring}[agent](question)
    if trace:
        write_trace(result)
    return result


def main() -> int:
    question = " ".join(sys.argv[1:]) or "What is the churn rate in snapshot A?"
    r = ask(question)
    print(f"[{r.agent} | {r.mode} | {r.latency_s:.2f}s]\n{r.answer}")
    for c in r.charts:
        print(f"chart: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
