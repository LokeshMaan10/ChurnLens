"""LLM mode: Claude with tool use over the same read-only tools (manual agentic loop).

Only used when an Anthropic credential and the model env var (configs/agents.yaml:
model_env_var) are set. The model id comes from that variable and is never hardcoded.
NOT exercised in this project's runs (no API key was provided); the offline agents are
what the evaluation measures.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import typing
from typing import Any

from churnlens.agents.base import AgentResult, ToolRecorder
from churnlens.config import load_config
from churnlens.tools import core

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_schema(name: str) -> dict[str, Any]:
    """Anthropic tool definition built from a core tool's signature and docstring."""
    fn = core.TOOLS[name]
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for pname, p in inspect.signature(fn).parameters.items():
        hint = hints.get(pname, str)
        base = next((t for t in typing.get_args(hint) if t is not type(None)), hint)
        props[pname] = {"type": JSON_TYPES.get(base, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"name": name, "description": inspect.getdoc(fn) or name,
            "input_schema": {"type": "object", "properties": props, "required": required}}


def run_claude(agent: str, system: str, tool_names: list[str], question: str) -> AgentResult:
    """Manual tool loop: call Claude, execute requested tools, return results, repeat."""
    import anthropic  # imported lazily so offline mode never needs the package configured

    cfg = load_config("agents")
    model = os.environ[cfg["model_env_var"]]
    client = anthropic.Anthropic()
    tools = ToolRecorder()
    defs = [tool_schema(n) for n in tool_names]
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentResult(agent, question, "", "llm")
    t0 = time.perf_counter()
    for _ in range(int(cfg["llm"]["max_tool_turns"])):
        resp = client.messages.create(
            model=model, max_tokens=int(cfg["llm"]["max_tokens"]), system=system,
            tools=defs, thinking={"type": "adaptive"},
            output_config={"effort": cfg["llm"]["effort"]}, messages=messages)
        result.input_tokens += resp.usage.input_tokens
        result.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            result.answer, result.refused = "The model declined this request.", True
            break
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        outputs = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = tools(block.name, **block.input)
                outputs.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out, default=str)})
            except Exception as e:  # report the refusal to the model; it can adapt
                outputs.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"{type(e).__name__}: {e}", "is_error": True})
        messages.append({"role": "user", "content": outputs})
    else:
        result.answer = "Stopped: too many tool turns."
    result.tool_calls = tools.calls
    result.sql = [c.output["sql_executed"] for c in tools.calls
                  if c.name == "run_sql" and c.output]
    result.charts = [c.output["path"] for c in tools.calls if c.name == "make_chart" and c.output]
    result.latency_s = time.perf_counter() - t0
    return result
