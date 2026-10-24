"""ChurnLens MCP server: exposes the read-only tool layer over the Model Context Protocol.

Run (stdio transport, for MCP clients such as Claude Desktop / Claude Code):
    python -m churnlens.mcp_server.server
Every tool is a thin wrapper over churnlens.tools.core, so the MCP surface, the agents
and the app share one tested implementation. No tool writes data or sends messages.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from churnlens.tools import core
from churnlens.tools.db import QueryRejected

EXPECTED_ERRORS = (QueryRejected, core.ToolError, KeyError, ValueError, TimeoutError)


def explained(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Turn expected refusals (non-read-only SQL, unknown customer, bad input) into MCP tool
    errors that carry the reason, instead of a generic failure."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except EXPECTED_ERRORS as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e
    return wrapper

# MCP SDK 2.x name for FastMCP. Every tool is annotated read-only.
mcp = MCPServer("churnlens")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@mcp.tool(annotations=READ_ONLY)
@explained
def run_sql(sql: str, max_rows: int = 200) -> dict[str, Any]:
    """Run ONE read-only SELECT over the curated views (see get_schema). Returns the exact SQL
    executed, columns and rows (capped at max_rows; `truncated` says if more matched).
    Writes, multiple statements and file-reading functions are rejected."""
    return core.run_sql(sql, max_rows)


@mcp.tool(annotations=READ_ONLY)
@explained
def get_schema(table: str | None = None) -> dict[str, Any]:
    """Describe the curated views (all, or one): purpose and column names/types."""
    return core.get_schema(table)


@mcp.tool(annotations=READ_ONLY)
@explained
def get_customer_profile(msno: str) -> dict[str, Any]:
    """Plan, payment, tenure, listening and RFM facts for one snapshot B customer (by msno)."""
    return core.get_customer_profile(msno)


@mcp.tool(annotations=READ_ONLY)
@explained
def get_risk_and_drivers(msno: str) -> dict[str, Any]:
    """Calibrated churn probability, risk band, quadrant and up to 5 risk-raising drivers
    with plain-language reason codes (associations, not causes)."""
    return core.get_risk_and_drivers(msno)


@mcp.tool(annotations=READ_ONLY)
@explained
def get_cltv(msno: str) -> dict[str, Any]:
    """12-month CLTV if the customer renews now (formula and survival methods), the value used
    for prioritisation, and expected value at risk. Margin is a placeholder assumption."""
    return core.get_cltv(msno)


@mcp.tool(annotations=READ_ONLY)
@explained
def recommend_action(msno: str) -> dict[str, Any]:
    """Rule-engine action for a customer with its cost, expected benefit and whether the
    current optimised contact list includes them (placeholder costs/save rates)."""
    return core.recommend_action(msno)


@mcp.tool(annotations=READ_ONLY)
@explained
def optimize_budget(budget_ntd: float, outreach_capacity: int, top_n: int = 20
                    ) -> dict[str, Any]:
    """Choose whom to contact within a budget (NTD) and a contact cap; returns the action mix,
    expected net benefit, its upper bound and the top of the list."""
    return core.optimize_budget(budget_ntd, outreach_capacity, top_n)


@mcp.tool(annotations=READ_ONLY)
@explained
def power_calc(baseline_rate: float, relative_reduction: float, alpha: float = 0.05,
               power: float = 0.8, treatment_share: float = 0.5) -> dict[str, Any]:
    """Users needed per arm to detect a relative churn reduction (two-sided z-test)."""
    return core.power_calc(baseline_rate, relative_reduction, alpha, power, treatment_share)


@mcp.tool(annotations=READ_ONLY)
@explained
def design_ab_test(n_selected: int, holdout_share: float, baseline_rate: float,
                   relative_reduction: float, alpha: float = 0.05,
                   power: float = 0.8) -> dict[str, Any]:
    """Split a selected list into contacted vs randomized holdout and check whether it can
    detect the assumed relative churn reduction (required sizes, achieved power)."""
    return core.design_ab_test(n_selected, holdout_share, baseline_rate, relative_reduction,
                               alpha, power)


@mcp.tool(annotations=READ_ONLY)
@explained
def run_stat_test(churned_control: int, n_control: int, churned_treatment: int,
                  n_treatment: int, alpha: float = 0.05,
                  planned_treatment_share: float | None = None) -> dict[str, Any]:
    """Two-proportion z-test (difference, CI, p-value); with planned_treatment_share also a
    sample-ratio-mismatch check."""
    return core.run_stat_test(churned_control, n_control, churned_treatment, n_treatment,
                              alpha, planned_treatment_share)


@mcp.tool(annotations=READ_ONLY)
@explained
def drift_report(top_n: int = 15) -> dict[str, Any]:
    """Latest drift results (PSI per feature and for predictions, data-quality flags,
    retraining verdict) comparing the training snapshot with the current one."""
    return core.drift_report(top_n)


@mcp.tool(annotations=READ_ONLY)
@explained
def make_chart(sql: str, kind: str, x: str, y: str, title: str) -> dict[str, Any]:
    """Run a read-only query and save a 'bar' or 'line' chart of column y by column x;
    returns the PNG path and the plotted data."""
    return core.make_chart(sql, kind, x, y, title)


def main() -> None:
    """Serve over stdio (default) or streamable HTTP (MCP_TRANSPORT=streamable-http, e.g.
    in docker-compose)."""
    import os
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
