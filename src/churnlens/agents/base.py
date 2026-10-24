"""Shared agent plumbing: results, tool-call recording, traces, guardrails.

Guardrails applied to every agent, in either mode:
  - tools are read-only (SQL guard in tools/db.py); no tool can send anything
  - causal-claim requests get an association-only answer (CAUSAL_DISCLAIMER)
  - every number in an answer can be checked against the tool outputs (grounding.py)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from churnlens.config import load_config, resolve_path
from churnlens.tools import core

CAUSAL_DISCLAIMER = ("ChurnLens data is observational: it shows associations with churn, "
                     "not cause and effect. Only a randomized experiment (see the campaign "
                     "planner) can measure what an action changes.")
CAUSAL_REQUEST = re.compile(r"\b(cause[sd]?|causal|why do(es)?|because|lead[s]? to|"
                            r"make[s]? (people|users|customers) churn|effect of|impact of|"
                            r"prove)\b", re.IGNORECASE)
WRITE_REQUEST = re.compile(r"\b(delete|drop|truncate|insert|update|alter|remove all|"
                           r"overwrite|modify the (data|table))\b", re.IGNORECASE)
SEND_REQUEST = re.compile(r"\b(send|email|text|sms|message them|notify|push to)\b",
                          re.IGNORECASE)
MSNO_IN_TEXT = re.compile(r"[A-Za-z0-9+/]{30,60}={0,2}")


@dataclass
class ToolCall:
    """One recorded tool invocation."""

    name: str
    args: dict[str, Any]
    output: Any
    seconds: float
    error: str | None = None


@dataclass
class AgentResult:
    """What an agent returns: the answer plus everything needed to audit it."""

    agent: str
    question: str
    answer: str
    mode: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)   # structured values for grading
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_calls"] = [{**asdict(c), "output": _short(c.output)} for c in self.tool_calls]
        return d


def _short(obj: Any, limit: int = 4000) -> Any:
    text = json.dumps(obj, default=str)
    return obj if len(text) <= limit else text[:limit] + "...(truncated)"


class ToolRecorder:
    """Calls tools by name and records every call (inputs, outputs, timing, errors)."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def __call__(self, name: str, **args: Any) -> Any:
        fn = core.TOOLS[name]
        t0 = time.perf_counter()
        try:
            out = fn(**args)
            self.calls.append(ToolCall(name, args, out, time.perf_counter() - t0))
            return out
        except Exception as e:  # recorded, then re-raised for the agent to handle
            self.calls.append(ToolCall(name, args, None, time.perf_counter() - t0,
                                       f"{type(e).__name__}: {e}"))
            raise


def resolve_mode(cfg: dict[str, Any]) -> str:
    """'llm' or 'offline' per configs/agents.yaml (auto: llm only if fully configured)."""
    mode = cfg["mode"]
    has_cred = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    has_model = bool(os.environ.get(cfg["model_env_var"]))
    if mode == "llm" and not (has_cred and has_model):
        raise RuntimeError(f"llm mode needs an Anthropic credential and {cfg['model_env_var']}")
    if mode == "auto":
        return "llm" if has_cred and has_model else "offline"
    return mode


def write_trace(result: AgentResult) -> None:
    """Append one JSON line per agent run (simple JSON-log tracing; no external service)."""
    cfg = load_config("agents")
    path = resolve_path(cfg["traces"]["path"], load_config("data"))
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **result.as_dict()}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def fmt_num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"
