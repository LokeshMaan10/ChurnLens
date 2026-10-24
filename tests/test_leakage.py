"""Leakage guards.

Two kinds of test:
  * unit tests of the snapshot SQL on tiny hand-built tables (always run)
  * data tests on the built snapshot and clean files (skipped until
    `python -m churnlens.snapshots.build` has run), which enforce, for every
    snapshot: each input's event timestamp is strictly before the cutoff, the
    scoring-only March logs never appear, and inputs cover only label users.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

import duckdb
import pytest

from churnlens.config import date_to_int, load_config, resolve_path
from churnlens.snapshots import build
from churnlens.snapshots.build import SNAPSHOT_INPUTS, TIMESTAMP_COLUMNS, scan, snapshot_input_sql

CUTOFF = dt.date(2017, 2, 1)


# ---------------------------------------------------------------- unit tests


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("CREATE TABLE lab (msno VARCHAR, is_churn TINYINT)")
    c.execute("INSERT INTO lab VALUES ('a', 0), ('b', 1)")
    c.execute("CREATE TABLE transactions (msno VARCHAR, transaction_date DATE, "
              "membership_expire_date DATE, is_cancel TINYINT)")
    c.execute("""INSERT INTO transactions VALUES
        ('a', DATE '2017-01-31', DATE '2017-02-28', 0),
        ('a', DATE '2017-02-01', DATE '2017-03-31', 0),   -- on the cutoff: excluded
        ('b', DATE '2017-02-10', DATE '2017-02-11', 1),   -- cancel inside label month
        ('x', DATE '2017-01-01', DATE '2017-02-01', 0)""")  # not a label user
    c.execute("CREATE TABLE user_logs (msno VARCHAR, date DATE, total_secs DOUBLE)")
    c.execute("INSERT INTO user_logs VALUES ('a', DATE '2017-01-31', 10), "
              "('a', DATE '2017-02-01', 20), ('x', DATE '2017-01-05', 30)")
    c.execute("CREATE TABLE user_logs_scoring (msno VARCHAR, date DATE, total_secs DOUBLE)")
    c.execute("INSERT INTO user_logs_scoring VALUES ('a', DATE '2017-03-05', 99)")
    c.execute("CREATE TABLE members (msno VARCHAR, registration_date DATE)")
    c.execute("INSERT INTO members VALUES ('a', DATE '2016-01-01'), ('b', DATE '2017-02-15')")
    return c


def test_transactions_and_cancels_inside_label_window_are_excluded(con) -> None:
    rows = con.execute(snapshot_input_sql("transactions", "lab", CUTOFF)).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("a", dt.date(2017, 1, 31))]


def test_logs_stop_before_cutoff_and_skip_scoring_logs(con) -> None:
    rows = con.execute(snapshot_input_sql("user_logs", "lab", CUTOFF)).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("a", dt.date(2017, 1, 31))]


def test_registration_after_cutoff_is_nulled_and_flagged(con) -> None:
    df = con.execute(snapshot_input_sql("members", "lab", CUTOFF)).df().set_index("msno")
    assert df.loc["b", "registered_after_cutoff"] and df["registration_date"].isna()["b"]
    assert not df.loc["a", "registered_after_cutoff"]


def test_snapshot_code_never_reads_scoring_only_data() -> None:
    source = inspect.getsource(build)
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#") and '"""' not in line)
    assert "user_logs_scoring" not in code.replace("`user_logs_scoring` is never", "")
    assert "scoring_only" not in code


# ---------------------------------------------------------------- data tests


def _snapshot_dirs() -> list[tuple[str, Path, dt.date]]:
    data_cfg = load_config("data")
    snap_cfg = load_config("snapshots")
    base = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    return [(name, base / name, spec["cutoff"]) for name, spec in snap_cfg["snapshots"].items()]


SNAPSHOTS = _snapshot_dirs()
built = pytest.mark.skipif(not all((d / "meta.json").exists() for _, d, _ in SNAPSHOTS),
                           reason="snapshots not built; run python -m churnlens.snapshots.build")


@built
@pytest.mark.parametrize(("name", "directory", "cutoff"), SNAPSHOTS)
def test_every_input_timestamp_is_before_cutoff(name: str, directory: Path,
                                                cutoff: dt.date) -> None:
    c = duckdb.connect()
    for input_name, column in TIMESTAMP_COLUMNS.items():
        path = directory / SNAPSHOT_INPUTS[input_name][0]
        late = c.execute(f"SELECT COUNT(*) FROM {scan(path)} "
                         f"WHERE {column} >= DATE '{cutoff.isoformat()}'").fetchone()[0]
        assert late == 0, f"{name}.{input_name}: {late} rows on/after cutoff {cutoff}"


@built
@pytest.mark.parametrize(("name", "directory", "cutoff"), SNAPSHOTS)
def test_scoring_only_logs_never_enter_snapshots(name: str, directory: Path,
                                                 cutoff: dt.date) -> None:
    data_cfg = load_config("data")
    scoring_spec = data_cfg["tables"]["user_logs_v2"]
    assert scoring_spec["scoring_only"]
    clean_dir = resolve_path(load_config("cleaning")["outputs"]["clean_dir"], data_cfg)
    scoring = clean_dir / "scoring_only" / "user_logs_v2"
    logs = directory / SNAPSHOT_INPUTS["user_logs"][0]
    assert "scoring_only" not in logs.as_posix()
    c = duckdb.connect()
    shared = c.execute(f"SELECT COUNT(*) FROM (SELECT msno, date FROM {scan(logs)} "
                       f"INTERSECT SELECT msno, date FROM {scan(scoring)})").fetchone()[0]
    assert shared == 0, f"{name}: {shared} snapshot log rows also in scoring-only logs"


@built
@pytest.mark.parametrize(("name", "directory", "cutoff"), SNAPSHOTS)
def test_inputs_cover_only_label_users(name: str, directory: Path, cutoff: dt.date) -> None:
    c = duckdb.connect()
    labels = scan(directory / "labels.parquet")
    for input_name in ("transactions", "user_logs", "members"):
        path = directory / SNAPSHOT_INPUTS[input_name][0]
        stray = c.execute(f"SELECT COUNT(*) FROM {scan(path)} WHERE msno NOT IN "
                          f"(SELECT msno FROM {labels})").fetchone()[0]
        assert stray == 0, f"{name}.{input_name}: {stray} rows for non-label users"


@built
@pytest.mark.parametrize(("name", "directory", "cutoff"), SNAPSHOTS)
def test_no_v2_restatements_in_snapshot_transactions(name: str, directory: Path,
                                                     cutoff: dt.date) -> None:
    """Under v1_then_v2, pre-cutoff history must come from the v1 export only."""
    rules = load_config("cleaning")
    if rules["transactions_source"] == "combined":
        pytest.skip("combined policy deliberately allows v2 rows inside v1's period")
    end = date_to_int(rules["v1_coverage_end"])
    c = duckdb.connect()
    n = c.execute(f"SELECT COUNT(*) FROM {scan(directory / 'transactions.parquet')} "
                  f"WHERE source <> 'v1' AND strftime(transaction_date, '%Y%m%d')::INT <= {end}"
                  ).fetchone()[0]
    assert n == 0


@built
@pytest.mark.parametrize(("name", "directory", "cutoff"), SNAPSHOTS)
def test_meta_matches_files(name: str, directory: Path, cutoff: dt.date) -> None:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["cutoff"] == cutoff.isoformat()
    for info in meta["inputs"].values():
        if "max_timestamp" in info and info["max_timestamp"] != "None":
            assert info["max_timestamp"] < cutoff.isoformat()
