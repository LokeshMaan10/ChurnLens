"""Grounding check: every number in an answer must come from a tool output (or the question).

A number in the answer is grounded if some number in the tool outputs (or the user's own
question) rounds to it at the precision shown, either as-is or as a percentage (x100).
Rank-like ordinals in list markers ("1.", "2.") are ignored.
"""

from __future__ import annotations

import json
import re
from typing import Any

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(%?)")
LIST_MARKER = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


def numbers_in_text(text: str) -> list[tuple[float, int, bool, str]]:
    """(value, decimals shown, is_percent, raw token) for each number in the text."""
    text = LIST_MARKER.sub(" ", text)
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group(0)
        pct = raw.endswith("%")
        token = raw.rstrip("%").replace(",", "")
        try:
            v = float(token)
        except ValueError:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        out.append((v, decimals, pct, raw))
    return out


def _collect(obj: Any, acc: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)
    elif isinstance(obj, str):
        for v, *_ in numbers_in_text(obj):
            acc.append(v)


def source_numbers(tool_outputs: list[Any], question: str = "") -> list[float]:
    """Every number appearing in the tool outputs (recursively) and in the question."""
    acc: list[float] = []
    for o in tool_outputs:
        _collect(o if not isinstance(o, str) else o, acc)
        if isinstance(o, dict):
            _collect(json.dumps(o, default=str), acc)
    _collect(question, acc)
    return acc


def is_grounded(value: float, decimals: int, pct: bool, sources: list[float]) -> bool:
    """True if some source number displays as `value` at this precision."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for s in sources:
        # A percentage may show a share (0.064 -> 6.4%) or a value already in percent.
        for cand in ((s * 100, s) if pct else (s,)):
            if abs(abs(cand) - abs(value)) <= tol:
                return True
    return False


def ungrounded(answer: str, tool_outputs: list[Any], question: str = "") -> list[str]:
    """Raw tokens of numbers in the answer that no tool output (or the question) supports."""
    sources = source_numbers(tool_outputs, question)
    return [raw for v, d, pct, raw in numbers_in_text(answer)
            if not is_grounded(v, d, pct, sources)]
