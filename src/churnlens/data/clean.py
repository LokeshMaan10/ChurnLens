"""Phase 1: clean the raw layer, combine transaction files, validate, build DuckDB views.

Run with `python -m churnlens.data.clean` (or `make clean-data`). Steps:
  1. validate raw Parquet against raw pandera contracts (batched)
  2. build clean tables with DuckDB SQL and write them to data/interim/{mode}/clean/
  3. validate clean Parquet against clean pandera schemas (batched) and assert
     table-wide key uniqueness in DuckDB
  4. create data/interim/{mode}/churnlens.duckdb with raw_* and clean views
  5. write the `cleaning` section of results.md and docs/cleaning_rules.md

`user_logs_v2` is cleaned with the same rules but written under scoring_only/
and exposed only as the `user_logs_scoring` view, never merged into `user_logs`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandera.pandas as pa
import pyarrow.dataset as ds

from churnlens.config import date_to_int, load_config, resolve_path
from churnlens.data.schemas import clean_schemas, raw_schemas
from churnlens.data.views import register_parquet_views, register_raw_views, table_source
from churnlens.reporting import fmt_int, fmt_pct, md_table, run_banner, write_section

TXN_VALUE_COLUMNS = ["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                     "actual_amount_paid", "is_auto_renew", "transaction_date",
                     "membership_expire_date", "is_cancel"]
# A restatement = same transaction on every field except the expiry date.
TXN_RESTATEMENT_KEY = [c for c in TXN_VALUE_COLUMNS if c != "membership_expire_date"]
LOG_COUNTS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]

# Clean table name -> (raw source view, schema kind, sub-path under clean_dir, partitioned)
CLEAN_TABLES: dict[str, tuple[str, str, str, bool]] = {
    "train": ("train", "labels", "train.parquet", False),
    "train_v2": ("train_v2", "labels", "train_v2.parquet", False),
    "members": ("members_v3", "members", "members.parquet", False),
    "transactions": ("transactions", "transactions", "transactions.parquet", False),
    "user_logs": ("user_logs", "logs", "user_logs", True),
    "user_logs_scoring": ("user_logs_v2", "logs", "scoring_only/user_logs_v2", True),
}

# Table-wide uniqueness that batched pandera validation cannot see.
KEY_ASSERTIONS: dict[str, list[str]] = {
    "train": ["msno"],
    "train_v2": ["msno"],
    "members": ["msno"],
    "transactions": TXN_VALUE_COLUMNS,
    "user_logs": ["msno", "date"],
    "user_logs_scoring": ["msno", "date"],
}


SOURCE_POLICY_TEXT = {
    "v1_then_v2": "all v1 rows, plus v2 rows dated after {end} (the period v1 does not cover)",
    "combined": "all v1 and all v2 rows",
    "v1_only": "v1 rows only",
}


@dataclass
class Rule:
    """A documented cleaning rule and how many rows it touched."""

    rule_id: str
    table: str
    action: str
    description: str
    rationale: str
    affected: int = 0
    denominator: int = 0


# ---------------------------------------------------------------- SQL builders


def ymd(col: str) -> str:
    """SQL: YYYYMMDD integer column -> DATE."""
    return f"strptime(CAST({col} AS VARCHAR), '%Y%m%d')::DATE"


def members_sql(rules: dict[str, Any]) -> str:
    """Clean members: implausible ages -> NULL, empty gender -> NULL, parsed registration date."""
    lo, hi = rules["plausible_age"]["min"], rules["plausible_age"]["max"]
    return f"""
        SELECT msno, city, bd AS bd_raw,
               CASE WHEN bd BETWEEN {lo} AND {hi} THEN bd END AS age,
               NULLIF(gender, '') AS gender, registered_via,
               {ymd('registration_init_time')} AS registration_date,
               NOT (bd BETWEEN {lo} AND {hi}) AS age_invalid
        FROM raw_members_v3"""


def transactions_union_sql(rules: dict[str, Any]) -> str:
    """Raw transaction rows tagged with their file, per the configured source policy."""
    source = rules["transactions_source"]
    v1 = "SELECT *, 'v1' AS src FROM raw_transactions"
    if source == "v1_only":
        return v1
    if source == "combined":
        return f"{v1} UNION ALL SELECT *, 'v2' AS src FROM raw_transactions_v2"
    if source == "v1_then_v2":
        # v2 contributes only the period v1 does not cover, so every row inside v1's
        # period (and so before both snapshot cutoffs) comes from the v1 export.
        end = date_to_int(rules["v1_coverage_end"])
        return (f"{v1} UNION ALL SELECT *, 'v2' AS src FROM raw_transactions_v2 "
                f"WHERE transaction_date > {end}")
    raise ValueError(f"unknown transactions_source: {source!r}")


def v2_rows_inside_v1_period(con: duckdb.DuckDBPyConnection, rules: dict[str, Any]) -> int:
    """Clean transaction rows taken from v2 although dated inside v1's coverage (leak guard)."""
    end = date_to_int(rules["v1_coverage_end"])
    return count(con, "SELECT COUNT(*) FROM transactions WHERE source IN ('v2', 'both') "
                      f"AND transaction_date <= {ymd(str(end))}")


def transactions_sql(rules: dict[str, Any]) -> str:
    """Combine, collapse exact duplicates, resolve v1/v2 restatements, parse dates."""
    cols = ", ".join(TXN_VALUE_COLUMNS)
    key = ", ".join(TXN_RESTATEMENT_KEY)
    drop_src = {"v2": "v1", "v1": "v2"}[rules["restatement_keep"]]
    min_date = date_to_int(rules["min_plausible_date"])
    return f"""
        WITH unioned AS ({transactions_union_sql(rules)}),
        exact AS (  -- identical rows (within or across files) kept once
            SELECT {cols},
                   CASE WHEN COUNT(DISTINCT src) = 2 THEN 'both' ELSE MIN(src) END AS source
            FROM unioned GROUP BY {cols}
        ),
        grouped AS (
            SELECT *,
                   BOOL_OR(source IN ('v1', 'both')) OVER w AS has_v1,
                   BOOL_OR(source IN ('v2', 'both')) OVER w AS has_v2,
                   COUNT(*) OVER w AS n_versions
            FROM exact WINDOW w AS (PARTITION BY {key})
        ),
        resolved AS (
            SELECT *, (has_v1 AND has_v2 AND n_versions > 1) AS restated
            FROM grouped
            WHERE NOT (has_v1 AND has_v2 AND n_versions > 1 AND source = '{drop_src}')
        )
        SELECT msno, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid,
               is_auto_renew, {ymd('transaction_date')} AS transaction_date,
               CASE WHEN membership_expire_date >= {min_date}
                    THEN {ymd('membership_expire_date')} END AS membership_expire_date,
               is_cancel, source, restated,
               membership_expire_date < {min_date} AS expire_implausible
        FROM resolved"""


def logs_sql(raw_view: str, rules: dict[str, Any]) -> str:
    """Clean logs: parsed date, total_secs capped at one day, cap flagged."""
    cap = float(rules["max_seconds_per_day"])
    counts = ", ".join(LOG_COUNTS)
    return f"""
        SELECT msno, {ymd('date')} AS date, {counts},
               LEAST(total_secs, {cap}) AS total_secs, total_secs > {cap} AS secs_capped,
               strftime({ymd('date')}, '%Y-%m') AS month
        FROM {raw_view}"""


def clean_table_sql(name: str, rules: dict[str, Any]) -> str:
    """SQL producing one clean table."""
    raw_view, kind, _, _ = CLEAN_TABLES[name]
    if kind == "labels":
        return f"SELECT msno, is_churn FROM raw_{raw_view}"
    if kind == "members":
        return members_sql(rules)
    if kind == "transactions":
        return transactions_sql(rules)
    return logs_sql(f"raw_{raw_view}", rules)


# ---------------------------------------------------------------- build + validate


def write_clean_table(con: duckdb.DuckDBPyConnection, name: str, rules: dict[str, Any],
                      clean_dir: Path) -> Path:
    """Materialise a clean table to Parquet (month-partitioned for logs). Returns its path."""
    _, _, sub_path, partitioned = CLEAN_TABLES[name]
    out = clean_dir / sub_path
    if out.is_dir():
        shutil.rmtree(out)
    elif out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    options = "FORMAT parquet, COMPRESSION zstd"
    if partitioned:
        options += ", PARTITION_BY (month)"
    con.execute(f"COPY ({clean_table_sql(name, rules)}) TO '{out.as_posix()}' ({options})")
    return out


def validate_parquet(path: Path, schema: pa.DataFrameSchema, batch_rows: int) -> int:
    """Validate a Parquet file/directory against a schema in batches. Returns rows validated."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    columns = [c for c in dataset.schema.names if c != "month"]
    rows = 0
    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        frame = batch.to_pandas(date_as_object=False)
        schema.validate(frame, lazy=True)  # raises SchemaErrors listing every failure
        rows += len(frame)
    return rows


def assert_unique(con: duckdb.DuckDBPyConnection, view: str, key: list[str]) -> int:
    """Number of duplicate rows on `key` (must be 0)."""
    k = ", ".join(key)
    return con.execute(f"SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT {k} "
                       f"FROM {view})) FROM {view}").fetchone()[0]


# ---------------------------------------------------------------- rules + counts


def count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a scalar COUNT query."""
    return int(con.execute(sql).fetchone()[0])


def documented_rules(con: duckdb.DuckDBPyConnection, rules: dict[str, Any]) -> list[Rule]:
    """Every cleaning rule with its affected-row count, from raw and clean views."""
    lo, hi = rules["plausible_age"]["min"], rules["plausible_age"]["max"]
    cap = rules["max_seconds_per_day"]
    min_date = date_to_int(rules["min_plausible_date"])
    n_members = count(con, "SELECT COUNT(*) FROM raw_members_v3")
    n_union = count(con, f"SELECT COUNT(*) FROM ({transactions_union_sql(rules)})")
    n_exact = count(con, f"SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(TXN_VALUE_COLUMNS)} "
                         f"FROM ({transactions_union_sql(rules)}))")
    n_txn = count(con, "SELECT COUNT(*) FROM transactions")
    n_logs = count(con, "SELECT COUNT(*) FROM user_logs")
    keep = rules["restatement_keep"]
    return [
        Rule("M1", "members", "set NULL", f"`age` = `bd` if {lo} ≤ bd ≤ {hi}, else NULL; "
             "raw value kept in `bd_raw`, `age_invalid` flags it",
             "Two thirds of `bd` values are 0 or negative; treating them as ages would be wrong.",
             count(con, "SELECT COUNT(*) FROM raw_members_v3 "
                        f"WHERE NOT (bd BETWEEN {lo} AND {hi})"),
             n_members),
        Rule("M2", "members", "set NULL", "empty-string `gender` → NULL",
             "Empty string means unknown; NULL makes that explicit.",
             count(con, "SELECT COUNT(*) FROM raw_members_v3 WHERE gender = ''"), n_members),
        Rule("D1", "all", "parse", "YYYYMMDD integers → DATE (`registration_date`, "
             "`transaction_date`, `membership_expire_date`, `date`)",
             "Typed dates make cutoff filters and date arithmetic safe. Raw contracts already "
             "reject unparseable values, so no rows are lost.", 0, 0),
        Rule("T1", "transactions", "select source",
             f"`transactions_source: {rules['transactions_source']}`: "
             + SOURCE_POLICY_TEXT[rules["transactions_source"]].format(
                 end=date_to_int(rules["v1_coverage_end"])),
             "v2 rows inside v1's period restate old transactions with later expiry dates, "
             "which can encode snapshot B's label (see the effect table below). The build "
             "fails if any v2 row inside v1's period reaches the clean table under v1_then_v2.",
             n_union, n_union),
        Rule("T2", "transactions", "drop",
             "exact duplicate rows (within or across files) kept once; "
             "`source` = v1 / v2 / both",
             "Identical rows are double-recorded; counting them twice inflates transaction counts.",
             n_union - n_exact, n_union),
        Rule("T3", "transactions", "drop",
             "restatements (same transaction on every field except expiry, "
             "present in both files): "
             f"keep the {keep} version, set `restated`",
             f"`restatement_keep: {keep}`. Keeping both would double-count one transaction.",
             n_exact - n_txn, n_union),
        Rule("T4", "transactions", "set NULL",
             f"`membership_expire_date` before {min_date} → NULL; "
             "`expire_implausible` flags it",
             "Values like 1970-01-01 are placeholders, not real expiries.",
             count(con, "SELECT COUNT(*) FROM transactions WHERE expire_implausible"), n_txn),
        Rule("T5", "transactions", "keep + document",
             "list price 0, plan days 0, paid > list price, and expiry before transaction date "
             "are kept unchanged",
             "Plausibly real (free trials, promotions, plan changes, cancellations). Features in "
             "Phase 5 decide how to use them.",
             count(con, "SELECT COUNT(*) FROM transactions WHERE plan_list_price = 0 OR "
                        "payment_plan_days = 0 OR actual_amount_paid > plan_list_price OR "
                        "membership_expire_date < transaction_date"), n_txn),
        Rule("L1", "user_logs", "cap", f"`total_secs` capped at {cap:,}; `secs_capped` flags it",
             "A day has 86,400 seconds; larger values are logging errors, but the row's play "
             "counts are still informative.",
             count(con, "SELECT COUNT(*) FROM user_logs WHERE secs_capped"), n_logs),
        Rule("L2", "user_logs", "separate",
             "`user_logs_v2` (March 2017) cleaned with the same rules but written to "
             "`scoring_only/` and exposed only as `user_logs_scoring`",
             "It overlaps snapshot B's label period; it must never become a training input.",
             count(con, "SELECT COUNT(*) FROM user_logs_scoring"),
             count(con, "SELECT COUNT(*) FROM user_logs_scoring")),
    ]


def source_impact(con: duckdb.DuckDBPyConnection, cutoff_b: int,
                  rules: dict[str, Any]) -> dict[str, int]:
    """How each v2 policy changes pre-cutoff transaction history for snapshot B."""
    pre = f"transaction_date < {cutoff_b}"
    v2_only = (f"FROM raw_transactions_v2 t WHERE {pre} AND NOT EXISTS (SELECT 1 FROM "
               "raw_transactions v WHERE v.msno = t.msno "
               "AND v.transaction_date = t.transaction_date)")
    month_after = date_to_int(_add_month(cutoff_b))
    latest = ("SELECT msno, MAX(membership_expire_date) AS e FROM ({src}) "
              f"WHERE {pre} GROUP BY msno")

    def changed_vs_v1(src: str) -> int:
        return count(con, f"""
            SELECT COUNT(*) FROM raw_train_v2 l
            JOIN ({latest.format(src='SELECT * FROM raw_transactions')}) a USING (msno)
            JOIN ({latest.format(src=src)}) b USING (msno)
            WHERE a.e IS DISTINCT FROM b.e""")

    configured = transactions_union_sql(rules).replace(", 'v1' AS src", "").replace(
        ", 'v2' AS src", "")
    return {
        "v2_rows_before_cutoff": count(
            con, f"SELECT COUNT(*) FROM raw_transactions_v2 WHERE {pre}"),
        "v2_only_rows_before_cutoff": count(con, f"SELECT COUNT(*) {v2_only}"),
        "v2_only_rows_expiring_after_label_month": count(
            con, f"SELECT COUNT(*) {v2_only} AND membership_expire_date >= {month_after}"),
        "b_label_users": count(con, "SELECT COUNT(*) FROM raw_train_v2"),
        "b_users_latest_expiry_changed": changed_vs_v1(
            "SELECT * FROM raw_transactions UNION ALL SELECT * FROM raw_transactions_v2"),
        "b_users_latest_expiry_changed_configured": changed_vs_v1(configured),
        "month_after_cutoff": month_after,
    }


def _add_month(yyyymmdd: int) -> str:
    """ISO date one calendar month after a YYYYMMDD date (day kept; fine for the 1st)."""
    y, m, d = yyyymmdd // 10000, yyyymmdd // 100 % 100, yyyymmdd % 100
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------- reporting


def render_section(data_cfg: dict[str, Any], rules_cfg: dict[str, Any], rules: list[Rule],
                   validation: dict[str, tuple[int, int]], dup_counts: dict[str, int],
                   impact: dict[str, int], cutoff_b: int) -> str:
    """Markdown for the `cleaning` section of results.md."""
    parts = ["## Phase 1: data layer and cleaning", "", run_banner(data_cfg), "",
             f"Transaction source policy: **{rules_cfg['transactions_source']}**; "
             f"restatements keep **{rules_cfg['restatement_keep']}**. "
             "Rule rationale: `docs/cleaning_rules.md`.", "", "### Rules applied", ""]
    parts.append(md_table(
        ["ID", "Table", "Action", "Rule", "Rows affected", "% of rows"],
        [[r.rule_id, r.table, r.action, r.description,
          fmt_int(r.affected) if r.denominator else "—",
          fmt_pct(r.affected, r.denominator) if r.denominator else "—"] for r in rules],
        "llllrr"))
    parts += ["", "### Validation", "",
              "pandera validates every row in batches; key uniqueness is asserted in DuckDB.", ""]
    parts.append(md_table(
        ["Table", "Raw rows validated", "Clean rows validated", "Key", "Duplicate keys"],
        [[t, fmt_int(raw_n), fmt_int(clean_n), f"({', '.join(KEY_ASSERTIONS[t])})"
          if t != "transactions" else "(all columns)", fmt_int(dup_counts[t])]
         for t, (raw_n, clean_n) in validation.items()], "lrrlr"))
    parts += ["", f"### Effect of including v2 on snapshot B history (cutoff {cutoff_b})", "",
              "Computed for both policies regardless of the setting above, so the trade-off "
              "stays visible.", ""]
    b = impact["b_label_users"]
    parts.append(md_table(["Measure", "Value"], [
        ["v2 rows dated before the cutoff", fmt_int(impact["v2_rows_before_cutoff"])],
        ["...with no same (msno, date) row in v1", fmt_int(impact["v2_only_rows_before_cutoff"])],
        [f"...of those, expiring on/after {impact['month_after_cutoff']}",
         f"{fmt_int(impact['v2_only_rows_expiring_after_label_month'])} "
         "(" + fmt_pct(impact["v2_only_rows_expiring_after_label_month"],
                       impact["v2_only_rows_before_cutoff"]) + ")"],
        ["Snapshot B label users", fmt_int(b)],
        ["...whose latest pre-cutoff expiry differs from v1 only, if v2 were used in full "
         "(`combined`)",
         f"{fmt_int(impact['b_users_latest_expiry_changed'])} "
         f"({fmt_pct(impact['b_users_latest_expiry_changed'], b)})"],
        [f"...same, under the configured policy (`{rules_cfg['transactions_source']}`)",
         f"{fmt_int(impact['b_users_latest_expiry_changed_configured'])} "
         f"({fmt_pct(impact['b_users_latest_expiry_changed_configured'], b)})"],
        ["Clean rows taken from v2 inside v1's period (must be 0 unless `combined`)",
         fmt_int(impact["v2_rows_inside_v1_period"])],
    ], "lr"))
    parts += ["", "An expiry beyond the label month is associated with the user not churning in "
              "that month, so v2 rows inside v1's period can carry label information into "
              "snapshot B features. `v1_then_v2` excludes them; the guard row above is enforced "
              "by the build."]
    return "\n".join(parts)


def render_rules_doc(rules_cfg: dict[str, Any], rules: list[Rule]) -> str:
    """docs/cleaning_rules.md, generated from the rule list."""
    lines = ["# Cleaning rules", "",
             "_Generated by `python -m churnlens.data.clean` from the rule list in "
             "`src/churnlens/data/clean.py`; parameters come from `configs/cleaning.yaml`. "
             "Row counts are in `reports/results.md`._", ""]
    for r in rules:
        lines += [f"## {r.rule_id}: {r.table} ({r.action})", "", f"- **Rule:** {r.description}",
                  f"- **Why:** {r.rationale}", ""]
    lines += ["## Global uniqueness", "",
              "Asserted in DuckDB after the build (batched pandera validation cannot see "
              "table-wide duplicates):", ""]
    lines += [f"- `{t}`: unique on ({', '.join(k)})" for t, k in KEY_ASSERTIONS.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Build, validate and document the clean layer."""
    data_cfg = load_config("data")
    rules_cfg = load_config("cleaning")
    cutoff_b = date_to_int(load_config("snapshots")["snapshots"]["B"]["cutoff"])
    out_cfg = rules_cfg["outputs"]
    clean_dir = resolve_path(out_cfg["clean_dir"], data_cfg)
    batch = int(rules_cfg["validation_batch_rows"])
    raw_contracts, clean_contracts = raw_schemas(), clean_schemas(rules_cfg)

    db_path = resolve_path(out_cfg["database"], data_cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # views only; always rebuilt from Parquet
    con = duckdb.connect(str(db_path))
    register_raw_views(con, data_cfg, prefix="raw_")

    validation: dict[str, tuple[int, int]] = {}
    clean_paths: dict[str, Path] = {}
    raw_validated: dict[str, int] = {}
    for name, (raw_name, kind, _, _) in CLEAN_TABLES.items():
        print(f"{name}: validating raw {raw_name}", flush=True)
        raw_path = table_source(data_cfg["tables"][raw_name], data_cfg)
        raw_validated[name] = validate_parquet(raw_path, raw_contracts[kind], batch)
        if name == "transactions" and rules_cfg["transactions_source"] != "v1_only":
            v2 = table_source(data_cfg["tables"]["transactions_v2"], data_cfg)
            raw_validated[name] += validate_parquet(v2, raw_contracts[kind], batch)
        print(f"{name}: building clean table", flush=True)
        clean_paths[name] = write_clean_table(con, name, rules_cfg, clean_dir)
        clean_n = validate_parquet(clean_paths[name], clean_contracts[kind], batch)
        validation[name] = (raw_validated[name], clean_n)

    register_parquet_views(con, clean_paths)
    dup_counts = {t: assert_unique(con, t, k) for t, k in KEY_ASSERTIONS.items()}
    bad = {t: n for t, n in dup_counts.items() if n}
    if bad:
        raise AssertionError(f"duplicate keys in clean tables: {bad}")

    leaked = v2_rows_inside_v1_period(con, rules_cfg)
    if leaked and rules_cfg["transactions_source"] != "combined":
        raise AssertionError(f"{leaked} clean transactions come from v2 inside v1's period")
    rules = documented_rules(con, rules_cfg)
    impact = {**source_impact(con, cutoff_b, rules_cfg), "v2_rows_inside_v1_period": leaked}
    con.close()

    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "cleaning", render_section(data_cfg, rules_cfg, rules, validation, dup_counts,
                                             impact, cutoff_b))
    resolve_path(out_cfg["rules_md"], data_cfg).write_text(render_rules_doc(rules_cfg, rules),
                                                           encoding="utf-8")
    print(f"clean layer + views written; database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
