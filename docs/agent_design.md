# Agent design

ChurnLens agents answer questions and draft plans on top of the pipeline outputs. They are
built so that **every number comes from a tool**, **nothing can be written or sent**, and
**relationships are described as associations, never causes**.

## Architecture

```mermaid
flowchart LR
    U[User / app chat / MCP client] --> S[Supervisor<br/>route()]
    S -->|analytics question| A[Analyst]
    S -->|customer id| C[Retention copilot]
    S -->|budget / campaign| P[Campaign planner]
    S -->|drift / retrain| M[Monitoring agent]
    A & C & P & M --> T[Tool layer<br/>src/churnlens/tools/core.py]
    T --> D[(Read-only DuckDB views<br/>over pipeline Parquet)]
    T -. same functions .-> MCP[MCP server<br/>src/churnlens/mcp_server/server.py]
```

- **Tool layer** (`tools/core.py`, Phase 11): 12 pure functions — `run_sql`, `get_schema`,
  `get_customer_profile`, `get_risk_and_drivers`, `get_cltv`, `recommend_action`,
  `optimize_budget`, `power_calc`, `design_ab_test`, `run_stat_test`, `drift_report`,
  `make_chart`. Agents, the MCP server and the Streamlit app all call these same functions.
- **MCP server** (`mcp_server/server.py`): the official Python MCP SDK (2.x `MCPServer`,
  stdio transport). Every tool is annotated `readOnlyHint=true`; expected refusals return an
  MCP tool error that states the reason.

## Two modes, one interface (`configs/agents.yaml`)

| Mode | When | How it answers |
|---|---|---|
| `offline` (default here) | no Anthropic credential or no `CHURNLENS_MODEL` | deterministic intent parsing + templates that call the tools; free; fully tested |
| `llm` | `ANTHROPIC_API_KEY` (or auth token) **and** `CHURNLENS_MODEL` set | Claude with tool use, manual agentic loop (`agents/llm.py`), adaptive thinking, effort from config |

The model id is read from the `CHURNLENS_MODEL` environment variable; it is never
hardcoded. **The LLM mode was not run in this project** (no API key was provided); Phase 13
measures the offline agents.

## Agents

| Agent | Input | Tools | Output |
|---|---|---|---|
| Analyst | natural-language question | `get_schema`, `run_sql`, `make_chart` | answer + the exact SQL it ran (+ chart on request); "can't answer offline" for unmapped questions |
| Retention copilot | a customer `msno` | profile, risk & drivers, CLTV, recommend_action | brief, recommended action, **draft** outreach marked "DRAFT ONLY — not sent" |
| Campaign planner | budget + capacity | `optimize_budget`, `design_ab_test` | target list, action mix, A/B design with randomized holdout and power check |
| Monitoring | scheduled (`make monitor`) or on request | `drift_report` | incident note separating data issues, structural (clock) shifts, calendar effects and behaviour shifts; retrain verdict |
| Supervisor | any request | — | routes to exactly one agent (rules offline; same rules as the LLM-mode fallback) |

## Guardrails (all enforced in code and tested)

1. **Read-only data.** `tools/db.py` allows exactly one statement of type `SELECT`, blocks
   file-reading table functions, caps rows and times out long queries. The connection is
   in-memory over Parquet views, so there is no database file to modify.
2. **No sending.** No tool can send, email or write; the copilot only drafts, and says so when
   asked to send.
3. **Grounding.** `agents/grounding.py` checks that every number in an answer appears in a
   tool output (or the user's question) at the precision shown, or as a percentage of a
   share. Phase 13 applies it to every answer.
4. **No causal claims.** Questions that ask about causes get the associational comparison
   plus a fixed disclaimer; answers avoid causal verbs.
5. **Tracing.** Every request is appended as one JSON line to
   `reports/agent_traces/traces.jsonl` (agent, mode, tool calls with inputs/outputs/timing,
   latency, tokens). Langfuse was not set up; JSON logs are the tracing backend.

## Prompts (LLM mode)

The system prompts live in `src/churnlens/agents/prompts.py`. All share these rules:

- every number must come from a tool result; never estimate;
- describe relationships as associations; explain that only experiments show causes;
- tools are read-only and nothing can be sent;
- money figures rest on placeholder costs, save rates and margins.

Each agent adds its role and the tools to call (analyst: `get_schema` then `run_sql`, show
SQL; copilot: four customer tools, draft only; planner: `optimize_budget` then
`design_ab_test`; monitoring: `drift_report`, separate data issues from behaviour shifts).
The supervisor prompt asks for a single agent name.

## Evaluation (Phase 13)

`src/churnlens/evals/golden_set.yaml` holds 45 analyst questions (with paraphrases) plus
copilot, planner, monitoring and routing checks. Expected answers are computed by
independently written reference SQL at eval time. Measures: routing, numeric accuracy (the
expected value appears in the answer), SQL accuracy (the agent's executed query returns the
reference result), grounding, latency, tokens, and a safety suite (8 write attempts, 3 send
attempts, 4 causal pushes). Results are in `reports/results.md` (agent_eval section).

Known limitation: the offline analyst is template-based, so it handles the intents it was
built for and says so for anything else. The golden set also served as a regression set
while building it (first run: 91.1% numeric, 84.3% grounded, 73.3% safety; all failures
were fixed and the causes are recorded in `docs/decisions.md`).
