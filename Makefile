# On Windows without `make`, run the command under each target directly.
PYTHON ?= .venv/Scripts/python

.PHONY: setup data audit clean-data snapshots eda rfm features model explain value prioritize experiments monitor agent-eval live-score exports docs app mcp pipeline lint test check

# Create the Python 3.11 venv, install the package with dev tools, enable hooks.
setup:
	py -3.11 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install

# Phase -1: filter downloaded archives to sampled Parquet + MANIFEST.md (idempotent).
data:
	$(PYTHON) -m churnlens.data.acquire --skip-missing

# Phase 0: profile raw tables -> reports/results.md (audit) + docs/data_dictionary.md.
audit:
	$(PYTHON) -m churnlens.analysis.audit

# Phase 1: validate, clean, combine transactions, build data/interim/{mode}/churnlens.duckdb.
clean-data:
	$(PYTHON) -m churnlens.data.clean

# Phase 2: snapshots A/B under data/processed/{mode}/snapshots + label verification.
snapshots:
	$(PYTHON) -m churnlens.snapshots.build

# Phase 3: EDA on snapshot A -> reports/figures, reports/eda, results.md.
eda:
	$(PYTHON) -m churnlens.analysis.eda

# Phase 4: RFM scores + segments (+ K-Means comparison) -> data/processed/{mode}/rfm.
rfm:
	$(PYTHON) -m churnlens.segmentation.run

# Phase 5: feature tables per snapshot -> data/processed/{mode}/features + docs.
features:
	$(PYTHON) -m churnlens.features.build

# Phase 6: tune/calibrate/threshold on A, evaluate on B, log to MLflow (~30 min small mode).
model:
	$(PYTHON) -m churnlens.modeling.run

# Phase 7: SHAP global/local explanations + top-5 drivers with reason codes (~15 min).
explain:
	$(PYTHON) -m churnlens.explain.run

# Phase 8: Cox survival model, formula + survival CLTV, revenue backtest.
value:
	$(PYTHON) -m churnlens.value.run

# Phase 9: value at risk, quadrants, rules engine, budget optimizer.
prioritize:
	$(PYTHON) -m churnlens.prioritize.run

# Phase 10: power table + SIMULATED A/B test and uplift demo.
experiments:
	$(PYTHON) -m churnlens.experiments.run

# Phase 14: drift (PSI + Evidently), retraining verdict, monitoring-agent incident note.
# Schedule this target (cron / Task Scheduler) after each new snapshot.
monitor:
	$(PYTHON) -m churnlens.monitoring.run

# Phase 13: golden-set, grounding and safety evaluation of the agents.
agent-eval:
	$(PYTHON) -m churnlens.evals.run

# Phase 15: score April 2017 users (scoring-only snapshot C), Power BI exports, app.
live-score:
	$(PYTHON) -m churnlens.scoring.live

exports:
	$(PYTHON) -m churnlens.scoring.exports

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

# Phase 11: MCP server over stdio.
mcp:
	$(PYTHON) -m churnlens.mcp_server.server

# Phase 16: README results section + interview walkthrough from saved outputs.
docs:
	$(PYTHON) -m churnlens.docsgen

# Full rebuild from raw Parquet, in order.
pipeline: audit clean-data snapshots eda rfm features model explain value prioritize experiments monitor live-score agent-eval exports docs

lint:
	$(PYTHON) -m ruff check src tests app

test:
	$(PYTHON) -m pytest -q

check: lint test
