"""Phase 2: build snapshots A and B, verify labels, report class balance.

Run with `python -m churnlens.snapshots.build` (or `make snapshots`).

For each snapshot this writes data/processed/{mode}/snapshots/{name}/:
  labels.parquet        msno, is_churn (as provided; never modified)
  transactions.parquet  clean transactions of label users, transaction_date < cutoff
  user_logs/            clean training logs of label users, date < cutoff (month-partitioned)
  members.parquet       members of label users; registration dates on/after the cutoff
                        are set NULL and flagged
  meta.json             cutoff, row counts, latest input dates

These directories are the ONLY sanctioned inputs for features (Phase 5). The
scoring-only March logs are never read here. tests/test_leakage.py enforces both.

Label verification is the one place post-cutoff transactions are read, because
it checks the labels themselves; its outputs never feed features.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from churnlens.config import load_config, resolve_path
from churnlens.reporting import fmt_int, fmt_pct, md_table, run_banner, write_section

# (file under the snapshot dir, partitioned by month)
SNAPSHOT_INPUTS: dict[str, tuple[str, bool]] = {
    "labels": ("labels.parquet", False),
    "transactions": ("transactions.parquet", False),
    "user_logs": ("user_logs", True),
    "members": ("members.parquet", False),
}
# Column holding each input's event timestamp, which must be strictly before the cutoff.
TIMESTAMP_COLUMNS: dict[str, str] = {
    "transactions": "transaction_date",
    "user_logs": "date",
    "members": "registration_date",
}


def snapshot_input_sql(name: str, labels_view: str, cutoff: dt.date) -> str:
    """SQL for one snapshot input, restricted to label users and to data before the cutoff."""
    c = f"DATE '{cutoff.isoformat()}'"
    users = f"msno IN (SELECT msno FROM {labels_view})"
    if name == "labels":
        return f"SELECT msno, is_churn FROM {labels_view}"
    if name == "transactions":
        return f"SELECT * FROM transactions WHERE {users} AND transaction_date < {c}"
    if name == "user_logs":
        # `user_logs` is the training log view; `user_logs_scoring` is never referenced.
        return (f"SELECT *, strftime(date, '%Y-%m') AS month FROM user_logs "
                f"WHERE {users} AND date < {c}")
    if name == "members":
        return (f"SELECT * EXCLUDE (registration_date), "
                f"CASE WHEN registration_date < {c} THEN registration_date END "
                f"AS registration_date, registration_date >= {c} AS registered_after_cutoff "
                f"FROM members WHERE {users}")
    raise ValueError(f"unknown snapshot input: {name}")


def write_input(con: duckdb.DuckDBPyConnection, sql: str, out: Path, partitioned: bool) -> None:
    """COPY a query to Parquet, replacing any previous output."""
    if out.is_dir():
        shutil.rmtree(out)
    elif out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    options = "FORMAT parquet, COMPRESSION zstd" + (", PARTITION_BY (month)" if partitioned else "")
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' ({options})")


def scan(path: Path) -> str:
    """read_parquet expression for a snapshot input file or partitioned directory."""
    if path.is_dir():
        return (f"(SELECT * EXCLUDE (month) FROM read_parquet('{path.as_posix()}/**/*.parquet', "
                "hive_partitioning = true))")
    return f"read_parquet('{path.as_posix()}')"


# ---------------------------------------------------------------- label verification


def verify_labels(con: duckdb.DuckDBPyConnection, labels_view: str, cutoff: dt.date,
                  window_days: int) -> dict[str, Any]:
    """Rebuild KKBox's churn rule from transactions and compare it with the provided labels.

    E = latest expiry over non-cancel transactions before the cutoff. Users whose E falls
    in the label month (cutoff .. cutoff + 1 month) are checked: "renewed" means a
    non-cancel transaction on/after the cutoff, within `window_days` of E, that extends
    the expiry beyond E. A check is "complete" when E + window_days is within the data.
    """
    c = f"DATE '{cutoff.isoformat()}'"
    month_end = f"({c} + INTERVAL 1 MONTH)"
    data_end = con.execute("SELECT MAX(transaction_date) FROM transactions").fetchone()[0]
    rows = con.execute(f"""
        WITH e AS (
            SELECT msno, MAX(membership_expire_date) AS expiry FROM transactions
            WHERE transaction_date < {c} AND is_cancel = 0 GROUP BY msno
        ),
        b AS (
            SELECT l.msno, l.is_churn, e.expiry,
                   CASE WHEN e.expiry IS NULL THEN 'no pre-cutoff expiry'
                        WHEN e.expiry < {c} THEN 'before label month'
                        WHEN e.expiry < {month_end} THEN 'in label month'
                        ELSE 'after label month' END AS position
            FROM {labels_view} l LEFT JOIN e USING (msno)
        )
        SELECT position, is_churn,
               EXISTS (SELECT 1 FROM transactions t WHERE t.msno = b.msno
                       AND t.transaction_date >= {c} AND t.is_cancel = 0
                       AND t.transaction_date <= b.expiry + {window_days}
                       AND t.membership_expire_date > b.expiry) AS renewed,
               b.expiry + {window_days} <= DATE '{data_end}' AS complete,
               COUNT(*) AS n
        FROM b GROUP BY ALL ORDER BY ALL""").fetchall()
    cells = [dict(zip(["position", "is_churn", "renewed", "complete", "n"], r, strict=True))
             for r in rows]
    in_month = [r for r in cells if r["position"] == "in label month"]

    def total(pred: Any) -> int:
        return sum(r["n"] for r in in_month if pred(r))

    positions: dict[str, list[int]] = {}
    for r in cells:
        n_churn = r["n"] if r["is_churn"] else 0
        prev = positions.get(r["position"], [0, 0])
        positions[r["position"]] = [prev[0] + r["n"], prev[1] + n_churn]
    return {
        "data_end": str(data_end),
        "positions": positions,
        "in_month": total(lambda r: True),
        "complete": total(lambda r: r["complete"]),
        "agree": total(lambda r: r["renewed"] != bool(r["is_churn"])),
        "agree_complete": total(lambda r: r["complete"] and r["renewed"] != bool(r["is_churn"])),
        "churn_but_renewed": total(lambda r: r["is_churn"] and r["renewed"]),
        "churners_in_month": total(lambda r: r["is_churn"]),
        "kept_but_no_renewal": total(lambda r: not r["is_churn"] and not r["renewed"]),
        "kept_but_no_renewal_complete": total(
            lambda r: not r["is_churn"] and not r["renewed"] and r["complete"]),
    }


# ---------------------------------------------------------------- build


def build_snapshot(con: duckdb.DuckDBPyConnection, name: str, spec: dict[str, Any],
                   out_dir: Path) -> dict[str, Any]:
    """Write one snapshot's inputs and return its metadata."""
    cutoff: dt.date = spec["cutoff"]
    labels_view = spec["labels"]
    meta: dict[str, Any] = {"snapshot": name, "cutoff": cutoff.isoformat(),
                            "labels": labels_view, "role": spec["role"], "inputs": {}}
    for input_name, (sub, partitioned) in SNAPSHOT_INPUTS.items():
        path = out_dir / sub
        write_input(con, snapshot_input_sql(input_name, labels_view, cutoff), path, partitioned)
        info: dict[str, Any] = {
            "path": sub,
            "rows": con.execute(f"SELECT COUNT(*) FROM {scan(path)}").fetchone()[0],
            "users": con.execute(f"SELECT COUNT(DISTINCT msno) FROM {scan(path)}").fetchone()[0],
        }
        ts = TIMESTAMP_COLUMNS.get(input_name)
        if ts:
            lo, hi = con.execute(f"SELECT MIN({ts}), MAX({ts}) FROM {scan(path)}").fetchone()
            info["min_timestamp"], info["max_timestamp"] = str(lo), str(hi)
        meta["inputs"][input_name] = info
    labels = scan(out_dir / "labels.parquet")
    meta["n_users"], meta["n_churn"] = con.execute(
        f"SELECT COUNT(*), SUM(is_churn) FROM {labels}").fetchone()
    meta["registered_after_cutoff"] = con.execute(
        f"SELECT COUNT(*) FROM {scan(out_dir / 'members.parquet')} "
        "WHERE registered_after_cutoff").fetchone()[0]
    meta["churn_rate_by_coverage"] = {
        label: con.execute(
            f"SELECT COUNT(*), SUM(is_churn) FROM {labels} l WHERE "
            f"{'' if has else 'NOT '}EXISTS (SELECT 1 FROM {scan(out_dir / sub)} x "
            "WHERE x.msno = l.msno)").fetchone()
        for label, sub, has in (
            ("with pre-cutoff logs", "user_logs", True),
            ("without pre-cutoff logs", "user_logs", False),
            ("without members row", "members.parquet", False),
        )
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def lookback_coverage(data_cfg: dict[str, Any], cutoff: dt.date,
                      windows: list[int]) -> dict[str, Any]:
    """Does the configured log window cover the longest feature lookback before this cutoff?"""
    longest = max(windows)
    needed = cutoff - dt.timedelta(days=longest)
    start = data_cfg["log_window_start"]
    return {"longest_window_days": longest, "needed_from": needed.isoformat(),
            "log_window_start": str(start), "covered": start <= needed,
            "incomplete_windows": [w for w in windows if start > cutoff - dt.timedelta(days=w)]}


# ---------------------------------------------------------------- reporting


def render_section(data_cfg: dict[str, Any], metas: dict[str, dict[str, Any]],
                   checks: dict[str, dict[str, Any]], coverage: dict[str, dict[str, Any]],
                   window_days: int) -> str:
    """Markdown for the `snapshots` section of results.md."""
    parts = ["## Phase 2: snapshots, labels, and leakage guards", "", run_banner(data_cfg), "",
             "### Class balance and input coverage", ""]
    rows = []
    for s, m in metas.items():
        i = m["inputs"]
        rows.append([s, m["role"], m["cutoff"], f"`{m['labels']}`", fmt_int(m["n_users"]),
                     fmt_int(m["n_churn"]), fmt_pct(m["n_churn"], m["n_users"]),
                     fmt_int(i["transactions"]["users"]), fmt_int(i["user_logs"]["users"]),
                     fmt_int(i["members"]["users"]), fmt_int(m["registered_after_cutoff"])])
    parts.append(md_table(["Snapshot", "Role", "Cutoff", "Labels", "Users", "Churners",
                           "Churn rate", "Users w/ txns", "Users w/ logs", "Users w/ members",
                           "Registered on/after cutoff"], rows, "llllrrrrrrr"))
    parts += ["", "Churn rate by input coverage (association only; coverage gaps are not causes):",
              ""]
    rows = [[s, k, fmt_int(n), fmt_pct(c or 0, n)]
            for s, m in metas.items() for k, (n, c) in m["churn_rate_by_coverage"].items()]
    parts.append(md_table(["Snapshot", "Group", "Users", "Churn rate"], rows, "llrr"))

    parts += ["", "### Leakage guards (latest timestamp in each snapshot input)", "",
              "Every input must end strictly before its cutoff; `tests/test_leakage.py` "
              "asserts this on the built files, and that the scoring-only March logs are absent.",
              ""]
    rows = [[s, n, info.get("max_timestamp", "—"), m["cutoff"],
             "yes" if info.get("max_timestamp", "") < m["cutoff"] else "**NO**"]
            for s, m in metas.items() for n, info in m["inputs"].items()
            if "max_timestamp" in info]
    parts.append(md_table(["Snapshot", "Input", "Latest timestamp", "Cutoff", "Before cutoff"],
                          rows, "lllll"))

    parts += ["", f"### Label verification (KKBox rule: no renewal within {window_days} days "
                  "of expiry)", "",
              "Labels are used as provided. This rebuilds the rule from transactions "
              "(post-cutoff rows allowed here only, never in features) to check the assumption.",
              ""]
    rows = [[s, pos, fmt_int(n), fmt_pct(c, n)]
            for s, chk in checks.items() for pos, (n, c) in sorted(chk["positions"].items())]
    parts.append(md_table(["Snapshot", "Latest pre-cutoff expiry", "Users", "Churn rate"],
                          rows, "llrr"))
    parts.append("")
    rows = []
    for s, chk in checks.items():
        rows.append([s, fmt_int(chk["in_month"]), fmt_pct(chk["agree"], chk["in_month"]),
                     f"{fmt_int(chk['complete'])} ({fmt_pct(chk['complete'], chk['in_month'])})",
                     fmt_pct(chk["agree_complete"], chk["complete"]),
                     f"{fmt_int(chk['churn_but_renewed'])} of {fmt_int(chk['churners_in_month'])}",
                     fmt_int(chk["kept_but_no_renewal"]), chk["data_end"]])
    parts.append(md_table(
        ["Snapshot", "Users w/ expiry in label month", "Agreement", "Window fully observed",
         "Agreement (fully observed)", "Labelled churn but renewal seen",
         "Labelled kept but no renewal seen", "Transactions end"], rows, "lrrrrrrl"))

    parts += ["", "### Log lookback coverage", ""]
    rows = [[s, c["longest_window_days"], c["needed_from"], c["log_window_start"],
             "yes" if c["covered"] else "**NO**",
             ", ".join(map(str, c["incomplete_windows"])) or "none"]
            for s, c in coverage.items()]
    parts.append(md_table(["Snapshot", "Longest window (days)", "Needs logs from",
                           "Log window starts", "Covered", "Windows with incomplete history"],
                          rows, "lrllll"))
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Build both snapshots, verify labels, and report."""
    data_cfg = load_config("data")
    snap_cfg = load_config("snapshots")
    windows = load_config("features")["listening_windows_days"]
    db_path = resolve_path(load_config("cleaning")["outputs"]["database"], data_cfg)
    base = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    con = duckdb.connect(str(db_path), read_only=True)

    metas, checks, coverage = {}, {}, {}
    for name, spec in snap_cfg["snapshots"].items():
        print(f"snapshot {name}: building", flush=True)
        metas[name] = build_snapshot(con, name, spec, base / name)
        checks[name] = verify_labels(con, spec["labels"], spec["cutoff"],
                                     int(snap_cfg["renewal_window_days"]))
        coverage[name] = lookback_coverage(data_cfg, spec["cutoff"], windows)
        if not coverage[name]["covered"]:
            print(f"WARNING snapshot {name}: log window misses windows "
                  f"{coverage[name]['incomplete_windows']}", flush=True)
    con.close()

    results = resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg)
    write_section(results, "snapshots", render_section(
        data_cfg, metas, checks, coverage, int(snap_cfg["renewal_window_days"])))
    (base / "label_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"snapshots written to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
